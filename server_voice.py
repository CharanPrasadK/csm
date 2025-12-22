import os
import io
import time
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from transformers import pipeline
from generator import load_csm_1b, Segment
from run_csm import SPEAKER_PROMPTS, prepare_prompt

app = FastAPI()

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONVERSATION_FILE = "conversation_log.wav"
MAX_CONTEXT = 3  # Keep last 3 turns for style consistency

# --- 1. Load Models ---
print(f"Loading Models on {DEVICE}...")

# A. Ears (Whisper) - Speech to Text
stt_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en", # Use 'base.en' or 'small.en' for better accuracy
    device=DEVICE
)

# B. Brain (Llama) - Text Generation
chat_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)

# C. Mouth (CSM) - Text to Speech
csm_generator = load_csm_1b(DEVICE)

print("All models loaded!")

# --- State Management ---
# We store the conversation state in memory
history_text = [
    {"role": "system", "content": "You are a helpful AI. Keep answers short and spoken naturally."}
]
history_audio_segments = [] # List of Segment objects for CSM context
full_audio_buffer = []      # List of tensors to save to disk

# Initialize with a voice prompt
start_prompt = SPEAKER_PROMPTS["conversational_a"]
initial_segment = prepare_prompt(
    start_prompt["text"], 0, start_prompt["audio"], csm_generator.sample_rate
)
history_audio_segments.append(initial_segment)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("voice.html", "r") as f:
        return f.read()

@app.post("/talk")
async def audio_talk(file: UploadFile = File(...)):
    global history_text, history_audio_segments, full_audio_buffer

    # --- Step 1: Process User Audio (STT) ---
    user_audio_bytes = await file.read()
    
    # Load audio for processing
    # Note: Whisper expects 16k, CSM expects 24k (usually). We handle resampling.
    user_tensor, user_sr = torchaudio.load(io.BytesIO(user_audio_bytes))
    user_tensor = user_tensor.squeeze(0) # (T,)

    # Transcribe (Save to temp file for Whisper pipeline simplicity, or pass bytes if handled)
    # For simplicity/speed with huggingface pipeline, passing numpy array:
    user_np = user_tensor.numpy()
    if user_sr != 16000:
        # Resample for Whisper if needed, but pipeline handles sampling usually if path provided.
        # Here we let the pipeline handle raw input if possible, or easiest: save temp
        import soundfile as sf
        sf.write("temp_input.wav", user_np, user_sr)
        result = stt_pipe("temp_input.wav")
        user_text = result["text"]
    
    print(f"User Said: {user_text}")
    
    # Add User to History
    history_text.append({"role": "user", "content": user_text})
    
    # Resample user audio for CSM Context (24k) & Saving
    if user_sr != csm_generator.sample_rate:
        user_tensor_csm = torchaudio.functional.resample(user_tensor, user_sr, csm_generator.sample_rate)
    else:
        user_tensor_csm = user_tensor

    # Create User Segment for CSM Context
    # We assign speaker=1 for User, speaker=0 for AI
    user_segment = Segment(text=user_text, speaker=1, audio=user_tensor_csm.to(DEVICE))
    history_audio_segments.append(user_segment)
    full_audio_buffer.append(user_tensor_csm.cpu())

    # --- Step 2: Generate Response (Brain) ---
    outputs = chat_pipe(
        history_text,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    ai_text = outputs[0]["generated_text"][-1]["content"]
    print(f"AI Thought: {ai_text}")
    history_text.append({"role": "assistant", "content": ai_text})

    # --- Step 3: Generate Audio (CSM) ---
    # Use sliding window context (Last N turns)
    recent_segments = [history_audio_segments[0]] + history_audio_segments[-MAX_CONTEXT:]
    
    ai_audio_tensor = csm_generator.generate(
        text=ai_text,
        speaker=0,
        context=recent_segments,
        max_audio_length_ms=30_000,
        temperature=0.6, # Lower temp for stability
        topk=20
    )
    
    # Update History
    ai_segment = Segment(text=ai_text, speaker=0, audio=ai_audio_tensor)
    history_audio_segments.append(ai_segment)
    full_audio_buffer.append(ai_audio_tensor.cpu())

    # --- Step 4: Save Conversation to Disk ---
    # Concatenate all audio so far and save
    conversation_tensor = torch.cat(full_audio_buffer, dim=0)
    torchaudio.save(CONVERSATION_FILE, conversation_tensor.unsqueeze(0), csm_generator.sample_rate)
    print(f"Updated {CONVERSATION_FILE}")

    # Return AI Audio to User
    buffer = io.BytesIO()
    torchaudio.save(buffer, ai_audio_tensor.unsqueeze(0).cpu(), csm_generator.sample_rate, format="wav")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    # Use HTTPS if possible for microphone permissions, but localhost works for http
    uvicorn.run(app, host="0.0.0.0", port=8000)