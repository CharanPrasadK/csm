import io
import os
import json
import torch
import torchaudio
import uvicorn
import time  # [!code ++] Added for timestamp generation
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, storage, db as firebase_db

# Load environment variables from .env file
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

# [!code ++] CHANGED: Switched to "read_speech_a.wav" for better narration
PROMPT_FILENAME = "prompts/read_speech_a.wav"

# [!code ++] CHANGED: Exact transcript for read_speech_a.wav
PROMPT_TEXT = (
    "And Lake turned round upon me, a little abruptly, his odd yellowish eyes, a little "
    "like those of the sea eagle, and the ghost of his smile that flickered on his "
    "singularly pale face, with a stern and insidious look, confronted me."
)

# --- Firebase Initialization ---
# Get credentials path and config from environment variables
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "serviceAccountKey.json")
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
FIREBASE_BUCKET = os.getenv("FIREBASE_BUCKET")

if not firebase_admin._apps:
    try:
        # Check if the credential file exists before initializing
        if os.path.exists(FIREBASE_CRED_PATH):
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DB_URL,
                'storageBucket': FIREBASE_BUCKET
            })
            print(f"Firebase Admin Initialized with {FIREBASE_CRED_PATH}")
        else:
            print(f"Warning: Firebase credentials file not found at {FIREBASE_CRED_PATH}")
    except Exception as e:
        print(f"Warning: Firebase Init Failed: {e}")


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
    bookId: str  # Required to update Firebase


def process_audio_task(text: str, book_id: str):
    """
    Background task to generate audio, upload it to Firebase Storage,
    and update the book document in Realtime Database.
    """
    global model_generator, FIXED_CONTEXT, FIXED_SPEAKER_ID
    
    print(f"[BG Task] Starting audio generation for Book ID: {book_id}")
    
    if model_generator is None:
        print("[BG Task] Error: Model not initialized.")
        return

    try:
        # --- DYNAMIC DURATION CALCULATION ---
        word_count = len(text.split())
        # Estimate: 2 words per second (generous) + 5 seconds buffer
        estimated_ms = int((word_count / 2.0) * 1000) + 5000
        
        # Safety Clamps: Minimum 10s, Maximum 3 minutes (180s)
        dynamic_max_length = max(10000, min(estimated_ms, 180000))
        
        print(f"[BG Task] Text length: {word_count} words. Setting max duration to: {dynamic_max_length}ms")

        # 1. Generate Audio
        audio_tensor = model_generator.generate(
            text=text,
            speaker=FIXED_SPEAKER_ID,
            context=FIXED_CONTEXT,
            max_audio_length_ms=dynamic_max_length
        )
        
        # 2. Save to In-Memory Buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), model_generator.sample_rate, format="wav")
        buffer.seek(0)
        
        # 3. Upload to Firebase Storage
        bucket = storage.bucket()
        blob_path = f"book-narratives/{book_id}.wav"
        blob = bucket.blob(blob_path)
        
        blob.upload_from_file(buffer, content_type="audio/wav")
        blob.make_public()
        
        # [!code ++] FIX: Add timestamp to URL to bust browser cache
        final_url = f"{blob.public_url}?t={int(time.time())}"
        
        # 4. Update Realtime Database
        ref = firebase_db.reference(f'books/{book_id}')
        ref.update({
            'narrativeAudioUrl': final_url
        })
        
        print(f"[BG Task] Success! Audio uploaded to: {final_url}")

    except Exception as e:
        print(f"[BG Task] FAILED: {e}")


@app.post("/generate")
async def generate_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Endpoint receives text and bookId.
    It returns immediately (202 Accepted) and processes audio in background.
    """
    # [!code ++] ADDED: Debug print to see exact text from frontend
    print(f"\n[API] Received Text for Book {request.bookId}:\n{request.text}\n")

    if model_generator is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    # Fire and Forget: Add generation to background queue
    background_tasks.add_task(process_audio_task, request.text, request.bookId)
    
    return JSONResponse(
        content={
            "status": "processing", 
            "message": "Audio generation started in background",
            "bookId": request.bookId
        }, 
        status_code=202
    )


if __name__ == "__main__":
    # Define Certificate Paths (mapped from host in deploy.sh)
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