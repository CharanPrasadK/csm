import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import re

class Brain:
    def __init__(self, device="cuda", model_id="meta-llama/Llama-3.2-1B-Instruct"):
        print(f"Loading Brain (LLM: {model_id})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # FIX: Explicitly set pad_token_id to silence warnings
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        self.history = [
            {"role": "system", "content": "You are a helpful AI assistant. Keep responses short, natural, and conversational."}
        ]

    def think_stream(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self.tokenizer.apply_chat_template(self.history, return_tensors="pt", add_generation_prompt=True).to(self.device)
        
        generation_kwargs = dict(
            input_ids=inputs, 
            streamer=streamer, 
            max_new_tokens=200, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id # FIX: Pass explicitly
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        full_response = ""
        
        for new_text in streamer:
            buffer += new_text
            full_response += new_text
            
            # Split by punctuation to stream sentences ASAP
            # Added more delimiters like ';' or '-' to reduce gaps for long sentences
            parts = re.split(r'([.?!\n;])', buffer)
            
            if len(parts) > 1:
                for i in range(0, len(parts) - 1, 2):
                    sentence = parts[i] + parts[i+1]
                    if sentence.strip():
                        yield sentence.strip()
                buffer = parts[-1]

        if buffer.strip():
            yield buffer.strip()
            
        self.history.append({"role": "assistant", "content": full_response})