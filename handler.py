"""
RunPod serverless handler for Qwen3-TTS voice cloning.

Accepts base64-encoded reference audio + transcript + text to synthesize.
Returns base64-encoded WAV audio.
"""

import io
import base64

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

    if not ref_audio or not ref_text:
        return {"error": "ref_audio (base64) and ref_text are required"}
    if not text:
        return {"error": "text is required"}

    # Decode base64 audio to numpy array (the library's base64 detection
    # fails on standard base64 containing '/' characters)
    raw_bytes = base64.b64decode(ref_audio)
    audio_np, audio_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    ref_audio_tuple = (audio_np, audio_sr)

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio_tuple,
        ref_text=ref_text,
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )

    return {
        "audio_base64": wav_to_base64(wavs[0], sr),
        "sample_rate": sr,
    }


runpod.serverless.start({"handler": handler})
