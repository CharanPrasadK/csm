import asyncio
import io
import torch
import torchaudio
import uvicorn
import warnings # FIX: Import warnings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from modules.stt import Ear
from modules.llm import Brain
from modules.tts import Mouth
from generator import Segment

# FIX: Suppress the annoying Whisper/Transformers warnings
warnings.filterwarnings("ignore")

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("--- Loading AI Models ---")
ear = Ear(device=DEVICE, model_id="openai/whisper-base")
brain = Brain(device=DEVICE)
mouth = Mouth(device=DEVICE)
print("--- System Ready ---")

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connected")
    
    current_task = None
    
    try:
        while True:
            message = await websocket.receive()
            
            # 1. INTERRUPT
            if "text" in message and message["text"] == "INTERRUPT":
                if current_task and not current_task.done():
                    print("🛑 Interrupted")
                    current_task.cancel()
                continue

            # 2. AUDIO INPUT
            if "bytes" in message:
                audio_data = message["bytes"]
                
                if current_task and not current_task.done():
                    current_task.cancel()
                
                current_task = asyncio.create_task(process_stream(websocket, audio_data))

    except WebSocketDisconnect:
        print("Disconnected")

async def process_stream(websocket: WebSocket, audio_data: bytes):
    try:
        # Listen
        user_tensor, sr = load_audio_from_bytes(audio_data)
        user_text = ear.listen(user_tensor.numpy(), sr)
        print(f"User: {user_text}")
        
        if not user_text.strip(): return

        # Context
        user_tensor_csm = torchaudio.functional.resample(user_tensor, sr, 24000)
        user_seg = Segment(text=user_text, speaker=1, audio=user_tensor_csm.to(DEVICE))

        # Stream
        for sentence in brain.think_stream(user_text):
            print(f"AI: {sentence}")
            
            # Generate
            audio_tensor = mouth.speak(sentence, user_seg)
            user_seg = None 
            
            # Send
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), 24000, format="wav")
            await websocket.send_bytes(buffer.getvalue())
            
            # REMOVED sleep() to reduce gaps between sentences
            # await asyncio.sleep(0.01) 

    except asyncio.CancelledError:
        print("--- Cancelled ---")
    except Exception as e:
        print(f"Error: {e}")

def load_audio_from_bytes(data: bytes):
    wav_file = io.BytesIO(data)
    tensor, sr = torchaudio.load(wav_file)
    return tensor.squeeze(0), sr

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)