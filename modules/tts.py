import torch
from generator import load_csm_1b, Segment
from run_csm import SPEAKER_PROMPTS, prepare_prompt

class Mouth:
    def __init__(self, device="cuda"):
        print("Loading Mouth (CSM 1B)...")
        self.device = device
        self.generator = load_csm_1b(device)
        self.sample_rate = self.generator.sample_rate
        
        # Initialize context
        start_prompt = SPEAKER_PROMPTS["conversational_a"]
        self.initial_segment = prepare_prompt(
            start_prompt["text"], 0, start_prompt["audio"], self.sample_rate
        )
        self.audio_context = [self.initial_segment]

    def speak(self, text: str, user_audio_segment=None) -> torch.Tensor:
        if user_audio_segment:
            self.audio_context.append(user_audio_segment)

        # Use last 3 segments for context to keep speed high
        recent_segments = [self.audio_context[0]] + self.audio_context[-3:]

        audio_tensor = self.generator.generate(
            text=text,
            speaker=0,
            context=recent_segments,
            max_audio_length_ms=10000, 
            temperature=0.6,
            topk=20
        )

        # FIX: Normalize Audio Volume (Boost low audio)
        max_val = torch.abs(audio_tensor).max()
        if max_val > 0:
            audio_tensor = audio_tensor / max_val * 0.9 # Normalize to 90% volume

        # Update context
        ai_segment = Segment(text=text, speaker=0, audio=audio_tensor)
        self.audio_context.append(ai_segment)

        return audio_tensor