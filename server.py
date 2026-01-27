import io
import os
import json
import torch
import torchaudio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv

load_dotenv()

# Import from project files
from models import Model, ModelArgs
from generator import Generator, Segment

# Global variables
model_generator = None
FIXED_CONTEXT = []
FIXED_SPEAKER_ID = 0

# --- Configuration ---
REPO_ID = "sesame/csm-1b"
PROMPT_FILENAME = "prompts/conversational_a.wav"
PROMPT_TEXT = (
    "like revising for an exam I'd have to try and like keep up the momentum because I'd "
    "start really early I'd be like okay I'm gonna start revising now and then like "
    "you're revising for ages and then I just like start losing steam I didn't do that "
    "for the exam we had recently to be fair that was a more of a last minute scenario "
    "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
    "sort of start the day with this not like a panic but like a"
)

def fix_load_csm_1b(device: str = "cuda") -> Generator:
    print(f"Downloading config from {REPO_ID}...")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    valid_keys = ModelArgs.__dataclass_fields__.keys()
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = ModelArgs(**filtered_config)

    print("Instantiating model architecture...")
    model = Model(config)

    print(f"Downloading weights from {REPO_ID}...")
    try:
        weights_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    except Exception:
        print("Safetensors not found or failed, trying bin...")
        weights_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(device=device, dtype=torch.bfloat16)
    return Generator(model)

def load_prompt_tensor(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_generator, FIXED_CONTEXT, FIXED_SPEAKER_ID
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Initializing CSM Model on {device}...")
    model_generator = fix_load_csm_1b(device=device)

    print("Loading voice prompt...")
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=PROMPT_FILENAME)
        audio_tensor = load_prompt_tensor(path, model_generator.sample_rate)
        segment = Segment(text=PROMPT_TEXT, speaker=FIXED_SPEAKER_ID, audio=audio_tensor)
        FIXED_CONTEXT = [segment]
        print("Voice context loaded.")
    except Exception as e:
        print(f"Error loading voice prompt: {e}")

    yield

app = FastAPI(lifespan=lifespan)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_audio(request: TTSRequest):
    global model_generator, FIXED_CONTEXT, FIXED_SPEAKER_ID
    if model_generator is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    try:
        print(f"Generating: '{request.text[:30]}...'")
        audio_tensor = model_generator.generate(
            text=request.text,
            speaker=FIXED_SPEAKER_ID,
            context=FIXED_CONTEXT,
            max_audio_length_ms=10000
        )
        audio_tensor = audio_tensor.unsqueeze(0).cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, model_generator.sample_rate, format="wav")
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Define Certificate Paths
    # Note: These are the paths INSIDE the container (mapped from your host)
    cert_path = "/etc/letsencrypt/live/csm-tts-bb.tribedemos.com/fullchain.pem"
    key_path = "/etc/letsencrypt/live/csm-tts-bb.tribedemos.com/privkey.pem"

    if os.path.exists(cert_path) and os.path.exists(key_path):
        print(f"Starting in HTTPS mode using certs at {cert_path}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=5006,
            ssl_certfile=cert_path,
            ssl_keyfile=key_path
        )
    else:
        print("Certificates not found. Starting in HTTP mode (UNSECURE).")
        uvicorn.run(app, host="0.0.0.0", port=5006)