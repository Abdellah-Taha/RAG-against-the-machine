from functools import lru_cache
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Llm:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 100):
        message = [
            {"role": "system", "content": (
                "You are a helpful assistant."
                "Answer the question using the contextual information provided."
            )},
            {"role": "user", "content": prompt}
        ]
        input_text = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True)
        output = self.model.generate(**input_text,
                                     max_length=max_length,
                                     use_cache=True,
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

@lru_cache()
def call_llm():
    return Llm()

def generate_response(prompt: str,context: List[str], max_length: int = 100):
    llm = call_llm()

    super_prompt = f"{prompt}\n\nContext:\n" + "\n".join(context) + "\nResponse:"

    return str(llm.generate(super_prompt, max_length))
