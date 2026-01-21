import os

import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

class LLMTemplate:
    def __init__(self):
        pass
        
    def __call__(self):
        raise NotImplementedError()
        

class LlamaInstruct(LLMTemplate):
    def __init__(self, name: str):
        super().__init__()
        self.model_name = name
        self.messages = [
            {
                "role": "system", 
                "content": "You are a warm and friendly chatbot who is always eager to help and offer kind words of support.",
            },
        ]

    def _clear_history(self):
        self.messages = [
            {
                "role": "system", 
                "content": "You are a warm and friendly chatbot who is always eager to help and offer kind words of support.",
            },
        ]
        
    def _load_weight(self):
        """load model and tokenizer from huggingface"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16, # 2 bytes
            device_map="auto",
        )
        
    def pipeline(self, prompt: str):
        self.messages.append(
            {
                "role": "user", 
                "content": prompt,
            }
        )

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        input_ids = inputs['input_ids']
        # 1.1* for llama, you need to set up terminators
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 2. run model inference and store full conversation
        input_output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id, # avoid warning
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        self.full_history = self.tokenizer.decode(input_output_ids[0], skip_special_tokens=False)
        # 2.1 decode manually
        output_ids = input_output_ids[0][input_ids.shape[-1]:]
        # 2.2 decode by tokenizer
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 3. add chat history
        self.messages.append(
            {
                "role": "assistant", 
                "content": answer,
            },
        )
        return answer


class QwenInstruct(LLMTemplate):
    def __init__(self, name: str):
        super().__init__()
        self.model_name = name
        self.messages = [
            {
                "role": "system", 
                "content": "You are an intelligent assistant capable of coordinating complex tasks and providing insightful reasoning.",
            },
        ]

    def _clear_history(self):
        self.messages = [
            {
                "role": "system", 
                "content": "You are a warm and friendly chatbot who is always eager to help and offer kind words of support.",
            },
        ]
        
    def _load_weight(self):
        """load model and tokenizer from huggingface"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16, # 2 bytes
            device_map="auto",
        )
        
    def pipeline(self, prompt: str):
        self.messages.append(
            {
                "role": "user", 
                "content": prompt,
            }
        )

        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with autocast("cuda"):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.messages.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        del model_inputs, generated_ids
        torch.cuda.empty_cache()
        return response


class GPTInstruct:
    def __init__(self, name: str):
        super().__init__()
        self.model_name = name
        if "gpt-4o-text-only" in name:
            self.model_name = "gpt-4o"
        self.messages = [
            {
                "role": "system", 
                "content": "You are an intelligent assistant capable of coordinating complex tasks and providing insightful reasoning.",
            },
        ]

    def _clear_history(self):
        self.messages = []
        
    def _load_weight(self):
        """load model and tokenizer from openai"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def pipeline(self, prompt: str) -> str:
        self.messages.append(
            {
                "role": "user", 
                "content": prompt,
            }
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=0, # temp. fixed at 0
        )
        response = response.choices[0].message.content

        self.messages.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response
    