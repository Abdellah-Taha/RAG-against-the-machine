from functools import lru_cache
from typing import List
from data_models import StudentSearchResults, MinimalSearchResults, MinimalSource
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llm:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens=150):
        message = [
            {"role": "system", "content": (
                "You are a helpful assistant."
                "Answer the question using the contextual information provided."
            )},
            {"role": "user", "content": prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False, 
            add_generation_prompt=True
        )
        input_text = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True)
        output = self.model.generate(**input_text,
                                     max_new_tokens=max_new_tokens,
                                     use_cache=True,
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

@lru_cache()
def call_llm():
    return Llm()

def generate_response(prompt: str,context: MinimalSearchResults, max_new_tokens: int = 150):
    llm = call_llm()
    #conver MinimalSource into a string then join it to the prompt
    result: List[str] = []
    for source in context.retrieved_sources:
        result.append("file_path: " + source.file_path +
                      " first_index: " + str(source.first_character_index) +
                      " last_index: " + str(source.last_character_index) +
                      "\n")
    super_prompt = f"{prompt}\n\nContext:\n" + "\n".join(result) + "\nResponse:"
    # print(super_prompt)
    return str(llm.generate(super_prompt, max_new_tokens))

def call_llm_foreach_query(queries: List[str], context: StudentSearchResults):
    responses = []
    for i, query in enumerate(queries):
        # print(context.search_results[i].retrieved_sources)
        responses.append(generate_response(query, context.search_results[i]))
    return responses
