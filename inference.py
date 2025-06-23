import io
import os
import pathlib
import sys
import base64
import logging
from threading import Thread
from typing import Iterable   # ← you used Iterable in the annotation

import torch
import soundfile as sf
import numpy as np
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines import ParlerEngine

# ───────────────────────────────────────────────────────────────
# CONFIGURE LOGGING
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ───────────────────────────────────────────────────────────────
# 1) LOAD MODEL & TOKENIZER
# ───────────────────────────────────────────────────────────────
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float32
logger.info(f"Using torch device '{torch_device}' with dtype {torch_dtype}")

env_model   = os.getenv("MODEL_ID", "").strip()
default_dir = pathlib.Path("/opt/ml/model")

if env_model:
    model_name = env_model
    logger.info(f"Using model specified via MODEL_ID env var: '{model_name}'")
elif (default_dir / "config.json").is_file():
    model_name = str(default_dir)
    logger.info("Using model found inside /opt/ml/model.")
else:
    model_name = "ai4bharat/indic-parler-tts"
    logger.info(f"/opt/ml/model empty – downloading '{model_name}' from the Hub.")

tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if is_flash_attn_2_available() and torch.cuda.is_available():
    attention_implementation = "flash_attention_2"
    logger.info("Using Flash Attention 2 backend.")
elif torch.__version__ >= "2.0" and torch.cuda.is_available():
    attention_implementation = "sdpa"
    logger.info("Flash Attention 2 unavailable – falling back to PyTorch SDPA.")
else:
    attention_implementation = "eager"
    logger.info("Using default (eager) attention implementation.")

try:
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attention_implementation,
    ).to(torch_device)
    model.eval()
except Exception:
    logger.exception(
        f"Failed to load model with {attention_implementation} attention, retrying with eager."
    )
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(torch_device)
    model.eval()

assert 0 <= tok.pad_token_id < model.config.vocab_size, "Invalid pad_token_id!"

t5_vocab_size = getattr(model.text_encoder.config, 'vocab_size', 32128)

# ───────────────────────────────────────────────────────────────
# 2) FLASK ROUTE FOR STREAMING RAW PCM AUDIO
# ───────────────────────────────────────────────────────────────
@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        data = request.get_json(force=True)
        text = data.get("prompt", "").strip()
        description = data.get("description", "").strip()
        language = data.get("language", "hi").strip() or "hi"
        voice_preset = data.get("voice_preset", "female calm 22-kHz studio").strip() or "female calm 22-kHz studio"
        comma_silence_duration = float(data.get("comma_silence_duration", 0.05))
        sentence_timeout = float(data.get("sentence_timeout", 0.30))

        if not text:
            return jsonify({"error": "No prompt provided"}), 400
        if not description:
            description = "A female speaker with a calm, clear voice."
        logger.info(f"Received prompt: {text}")
        logger.info(f"Received description: {description}")
        logger.info(f"Language: {language}, Voice preset: {voice_preset}")

        # Create a per-request ParlerEngine for custom params
        engine = ParlerEngine(
            model_name = "ai4bharat/indic-parler-tts",
            language    = language,
            voice_preset= voice_preset,
            attn_implementation="flash_attention_2",
            torch_dtype = torch.float16,
            device_map  = "auto"
        )

        stream = TextToAudioStream(
            engine,
            comma_silence_duration = comma_silence_duration,
            sentence_timeout       = sentence_timeout
        )
        # For Parler, feed both description and prompt
        stream.feed({"prompt": text, "description": description})
        # Get frame rate from engine config if available
        try:
            frame_rate = engine.model.config.sampling_rate
        except Exception:
            frame_rate = 22050  # fallback default

        def generate_pcm_chunks():
            for audio_chunk in stream:
                audio_int16 = (audio_chunk * 32767).astype('int16')
                yield audio_int16.tobytes()

        headers = {
            "Transfer-Encoding": "chunked",
            "Content-Type": f"audio/L16; rate={frame_rate}; channels=1",
        }
        return Response(generate_pcm_chunks(), headers=headers, direct_passthrough=True)
    except Exception as e:
        logger.exception("Error during TTS synthesis in /invocations")
        return jsonify({"error": f"TTS synthesis failed: {str(e)}"}), 500

# ───────────────────────────────────────────────────────────────
# 3) FLASK ROUTE FOR HEALTH-CHECK
# ───────────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ───────────────────────────────────────────────────────────────
# 4) RUN THE FLASK APP (only when launched directly)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
