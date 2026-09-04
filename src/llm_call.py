from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def call_llm(context, question):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    prompt = "..."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,       # Maximum length of the generated response
        do_sample=True,           # Enables creative/random generation
        temperature=0.7,          # Controls creativity (lower is more focused)
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(response)