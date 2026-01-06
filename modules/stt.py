import torch
from transformers import pipeline
import numpy as np

class Ear:
    def __init__(self, device="cuda", model_id="openai/whisper-base"): 
        print(f"Loading Ear (STT: {model_id})...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

    def listen(self, audio_data: np.ndarray, sample_rate=16000) -> str:
        try:
            # FORCE ENGLISH: This stops the Korean/Japanese hallucinations
            result = self.pipe(
                {"raw": audio_data, "sampling_rate": sample_rate},
                generate_kwargs={"language": "en"} 
            )
            text = result["text"].strip()
            return text
        except Exception as e:
            print(f"STT Error: {e}")
            return ""