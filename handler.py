"""
RunPod serverless handler for Qwen3-TTS voice cloning.

Accepts base64-encoded reference audio + transcript + text to synthesize.
Returns base64-encoded WAV audio.
"""

import io
import base64
import time

import numpy as np
import soundfile as sf
import torch
import runpod
from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Model (loaded once at startup, persists across requests)
# ---------------------------------------------------------------------------

MODEL_PATH = "/model"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE, attn_implementation="flash_attention_2"
)
print("Model ready.")


def wav_to_base64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def handler(job):
    inp = job["input"]

    ref_audio = inp.get("ref_audio")
    ref_text = inp.get("ref_text")
    text = inp.get("text")
    language = inp.get("language", "English")

    # Generation parameters (all optional, sensible defaults)
    max_new_tokens = inp.get("max_new_tokens", 2048)
    temperature = inp.get("temperature", 0.9)
    top_k = inp.get("top_k", 50)
    top_p = inp.get("top_p", 1.0)
    repetition_penalty = inp.get("repetition_penalty", 1.05)

    if not ref_audio or not ref_text:
        return {"error": "ref_audio (base64) and ref_text are required"}
    if not text:
        return {"error": "text is required"}

    t0 = time.time()

    # --- Decode reference audio ---
    t_dec_start = time.time()
    raw_bytes = base64.b64decode(ref_audio)
    audio_np, audio_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    ref_audio_tuple = (audio_np, audio_sr)
    print(f"[timing] Audio decode: {time.time() - t_dec_start:.3f}s")

    # --- Clone voice (encode ref audio into reusable prompt) ---
    t_clone_start = time.time()
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_tuple,
        ref_text=ref_text,
    )
    torch.cuda.synchronize()
    print(f"[timing] Voice clone (prompt creation): {time.time() - t_clone_start:.3f}s")

    # --- Generate speech from cloned voice ---
    t_gen_start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=top_k,
        subtalker_top_p=top_p,
        subtalker_temperature=temperature,
    )
    torch.cuda.synchronize()
    t_generate = time.time() - t_gen_start
    audio_duration = len(wavs[0]) / sr
    print(f"[timing] Generation: {t_generate:.3f}s | Audio duration: {audio_duration:.2f}s | RTF: {t_generate / audio_duration:.2f}x")

    print(f"[timing] Total: {time.time() - t0:.3f}s")

    return {
        "audio_base64": wav_to_base64(wavs[0], sr),
        "sample_rate": sr,
    }


runpod.serverless.start({"handler": handler})
