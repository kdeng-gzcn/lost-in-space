import os
from platform import processor
from typing import List, Any, Tuple, Dict, Union
import logging
import io
import base64

import torch
from PIL import Image

from openai import OpenAI
import anthropic

from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    GenerationConfig, # for Phi4VisionInstruct
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoModelForImageTextToText,
    Gemma3ForConditionalGeneration,
    AutoModelForVision2Seq,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    Glm4vForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images


class VLMTemplate:
    def __init__(self, model_id: str) -> None:
        self.logger = logging.getLogger(__name__) # setup logger

        self.model_id = model_id
        self._load_weight() # setup model and processor
    
    def _load_weight(self) -> None:
        raise NotImplementedError
    
    def format_msg(
            self,
            images: List[Image.Image],
            prompt: str,
        ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return messages

    def qa(
            self,
            images: List[Image.Image],
            messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
            max_new_tokens: int = 1024
        ) -> str:
        """ No batch support for now """
        raise NotImplementedError


## Open source models (multi image with text input)
class BLIPVisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)


class LlavaNextVisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        
    def _load_weight(self) -> None:
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id, use_fast=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def qa(
            self, 
            images: List[Image.Image], 
            messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], 
            max_new_tokens: int = 1024
        ) -> str:
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=images, 
            text=text,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        response = self.processor.decode(
            output_ids[:, inputs["input_ids"].shape[-1]:][0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True,
        )
        response = response.strip()
        return response


class LlavaOneVisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        
    def _load_weight(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto",
        )

    def qa(
            self, 
            images: List[Image.Image], 
            messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], 
            max_new_tokens: int = 1024
        ) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            )
        
        response = self.processor.decode(
            output_ids[:, inputs["input_ids"].shape[-1]:][0], 
            skip_special_tokens=True, 
        )
        response = response.strip()
        return response


class Idefics3VisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)

    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def format_msg(
            self,
            images: List[Image.Image],
            prompt: str,
        ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return messages

    def qa(
            self, 
            images: List[Image.Image], 
            messages,
            max_new_tokens: int = 1024
        ) -> str:
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text, 
            images=images, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        return response


class DeepseekVisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)

    # def _load_weight(self):
    #     self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
    #     self.tokenizer = self.processor.tokenizer
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.model_name, 
    #         trust_remote_code=True,
    #     )
    #     self.model = self.model.to(torch.bfloat16).cuda().eval()


class Phi3VisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            num_crops=16,
            use_fast=True,
        ) 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype="auto",
            _attn_implementation='flash_attention_2' # falsh-attn needed   
        )


class Phi4VisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)

    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).cuda()
        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(
            self.model_id
        )


class Llama4VisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)

    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16, # 2 bytes
        )


class Gemma3VisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()


class QwenVisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, 
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
    def qa(self, 
           images: List[Image.Image],
           messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
           max_new_tokens: int = 1024
        ) -> str:
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, add_vision_id=True,
        ) # important to have (add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        return response
    

class Qwen3VisionInstruct(VLMTemplate):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        G = Qwen3VLMoeForConditionalGeneration if "A3B" in self.model_id else Qwen3VLForConditionalGeneration
        self.model = G.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )
        
    def qa(self, 
           images: List[Image.Image],
           messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
           max_new_tokens: int = 1024
        ) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        return response


class VLMR1VisionInstruct(VLMTemplate):
    def __init__(self, model_id):
        super().__init__(model_id=model_id)
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
    def qa(self, 
           images: List[Image.Image], 
           messages: str,
           max_new_tokens: int = 4096
        ) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return response


class GLMVisionInstruct(VLMTemplate):
    def __init__(self, model_id):
        super().__init__(model_id=model_id)
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
    def qa(self, 
           images: List[Image.Image], 
           messages: str,
           max_new_tokens: int = 4096
       ) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        return response
    
    
## Proprietary models
class GPTVisionInstruct(VLMTemplate):
    def __init__(self, model_id):
        super().__init__(model_id=model_id)
        self.prompt_tokens = []
        self.completion_tokens = []
        
    def _load_weight(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _calculate_input_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "gpt-4o-mini": 0.15, # 0.15$ per million tokens
            "gpt-4o": 2.5, # 2.5$ per million tokens
            "gpt-4-turbo": 10, # 10$ per million tokens
            "gpt-5": 1.25, # 1.25$ per million tokens
        }
        assert self.model_id in cost_map, self.logger.error(f"Model {self.model_id} not found in input cost map")
        return cost_map[self.model_id] * num_tokens / 1e6
    
    def _calculate_output_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "gpt-4o-mini": 0.6, # 0.15$ per million tokens
            "gpt-4o": 10, # 2.5$ per million tokens
            "gpt-4-turbo": 30, # 30$ per million tokens
            "gpt-5": 10, # 10$ per million tokens
        }
        assert self.model_id in cost_map, self.logger.error(f"Model {self.model_id} not found in output cost map")
        return cost_map[self.model_id] * num_tokens / 1e6

    def _print_tokens_usage(self, completiion: Any) -> None:
        """not used"""
        self.logger.info(f"🤡 Prompt Tokens Usage: {completiion.usage.prompt_tokens}")
        self.logger.info(f"👾 Prompt Tokens Cost: {self._calculate_input_tokens_cost(completiion.usage.prompt_tokens)}")
        self.logger.info(f"🤡 Completion Tokens Usage: {completiion.usage.completion_tokens}")
        self.logger.info(f"🤡 Completion Reasoning Tokens Usage: {completiion.usage.completion_tokens_details.reasoning_tokens}")
        self.logger.info(f"👾 Completion Tokens Cost: {self._calculate_output_tokens_cost(completiion.usage.completion_tokens)}")
        self.logger.info(f"🤡 Total Tokens Usage: {completiion.usage.total_tokens}")

    def _collect_completions(self, completiion: Any) -> None:
        self.prompt_tokens.append(completiion.usage.prompt_tokens)
        self.completion_tokens.append(completiion.usage.completion_tokens)

    def print_total_tokens_usage(self) -> None:
        self.logger.info(f"😚💦💬 Total Prompt Tokens Usage: {sum(self.prompt_tokens)}")
        self.logger.info(f"💰 Total Prompt Tokens Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens))}")
        self.logger.info(f"🤖💬 Total Completion Tokens Usage: {sum(self.completion_tokens)}")
        self.logger.info(f"💰 Total Completion Tokens Cost: {self._calculate_output_tokens_cost(sum(self.completion_tokens))}")
        self.logger.info(f"🤡🤡🤡 Total Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)) + self._calculate_output_tokens_cost(sum(self.completion_tokens))}")

    def _pil_to_base64(self, pil_image, format="JPEG") -> str:
        """Convert a PIL Image to a base64-encoded string."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def format_msg(
            self,
            images: List[Image.Image],
            prompt: str,
        ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """
        Rewrite to fit OpenAI API format
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._pil_to_base64(img)}",
                            },
                        } for img in images
                    ],
                ]
            }
        ]
        return messages

    def qa(
        self,
        images: Tuple[Image.Image], 
        messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
        max_new_tokens: int = 1024
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0., # temp. fixed at 0
        )
        self._collect_completions(completion)
        response = completion.choices[0].message.content
        return response
    

class GPT5(VLMTemplate):
    def __init__(self, model_id):
        super().__init__(model_id=model_id)
        self.prompt_tokens = []
        self.completion_tokens = []
        
    def _load_weight(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _calculate_input_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "gpt-4o-mini": 0.15, # 0.15$ per million tokens
            "gpt-4o": 2.5, # 2.5$ per million tokens
            "gpt-4-turbo": 10, # 10$ per million tokens
            "gpt-5": 1.25, # 1.25$ per million tokens
        }
        assert self.model_id in cost_map, self.logger.error(f"Model {self.model_id} not found in input cost map")
        return cost_map[self.model_id] * num_tokens / 1e6
    
    def _calculate_output_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "gpt-4o-mini": 0.6, # 0.15$ per million tokens
            "gpt-4o": 10, # 2.5$ per million tokens
            "gpt-4-turbo": 30, # 30$ per million tokens
            "gpt-5": 10, # 10$ per million tokens
        }
        assert self.model_id in cost_map, self.logger.error(f"Model {self.model_id} not found in output cost map")
        return cost_map[self.model_id] * num_tokens / 1e6

    def _print_tokens_usage(self, completiion: Any) -> None:
        """not used"""
        self.logger.info(f"🤡 Prompt Tokens Usage: {completiion.usage.prompt_tokens}")
        self.logger.info(f"👾 Prompt Tokens Cost: {self._calculate_input_tokens_cost(completiion.usage.prompt_tokens)}")
        self.logger.info(f"🤡 Completion Tokens Usage: {completiion.usage.completion_tokens}")
        self.logger.info(f"🤡 Completion Reasoning Tokens Usage: {completiion.usage.completion_tokens_details.reasoning_tokens}")
        self.logger.info(f"👾 Completion Tokens Cost: {self._calculate_output_tokens_cost(completiion.usage.completion_tokens)}")
        self.logger.info(f"🤡 Total Tokens Usage: {completiion.usage.total_tokens}")

    def _collect_completions(self, completiion: Any) -> None:
        self.prompt_tokens.append(completiion.usage.prompt_tokens)
        self.completion_tokens.append(completiion.usage.completion_tokens)

    def print_total_tokens_usage(self) -> None:
        self.logger.info(f"😚💦💬 Total Prompt Tokens Usage: {sum(self.prompt_tokens)}")
        self.logger.info(f"💰 Total Prompt Tokens Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens))}")
        self.logger.info(f"🤖💬 Total Completion Tokens Usage: {sum(self.completion_tokens)}")
        self.logger.info(f"💰 Total Completion Tokens Cost: {self._calculate_output_tokens_cost(sum(self.completion_tokens))}")
        self.logger.info(f"🤡🤡🤡 Total Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)) + self._calculate_output_tokens_cost(sum(self.completion_tokens))}")

    def _pil_to_base64(self, pil_image, format="JPEG") -> str:
        """Convert a PIL Image to a base64-encoded string."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def format_msg(
            self,
            images: List[Image.Image],
            prompt: str,
        ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """
        Rewrite to fit OpenAI API format
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    *[
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{self._pil_to_base64(img)}",
                        } for img in images
                    ],
                ]
            }
        ]
        return messages

    def qa(
        self,
        images: Tuple[Image.Image], 
        messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
        max_new_tokens: int = 1024
    ) -> str:
        # completion = self.client.chat.completions.create(
        #     model=self.model_id,
        #     messages=messages,
        #     max_completion_tokens=max_new_tokens,
        # )
        # self._collect_completions(completion)
        # response = completion.choices[0].message.content
        response = self.client.responses.create(
            model="gpt-5",
            input=messages,
        )
        response = response.output_text
        return response


class AnthropicVisionInstruct(VLMTemplate):
    def __init__(self, model_id):
        super().__init__(model_id=model_id)
        self.prompt_tokens = []
        self.output_tokens = []
        
    def _load_weight(self) -> None:
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _calculate_input_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "claude-sonnet-4-20250514": 3, # 3$ per million tokens
            "claude-opus-4-20250514": 15, # 15$ per million tokens
        }
        assert self.model_id in cost_map, self.logger.error(f"Model {self.model_id} not found in input cost map")
        return cost_map[self.model_id] * num_tokens / 1e6
    
    def _calculate_output_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "claude-sonnet-4-20250514": 15, # 15$ per million tokens
            "claude-opus-4-20250514": 75, # 75$ per million tokens
        }
        assert self.model_id in cost_map, self.logger.error(f"Model {self.model_id} not found in output cost map")
        return cost_map[self.model_id] * num_tokens / 1e6

    def _collect_tokens_count(self, response: str, messages: List[Dict[str, str]]) -> None:
        input_tokens = self.client.messages.count_tokens(
            model=self.model_id,
            messages=messages,
        )
        self.prompt_tokens.append(input_tokens.input_tokens)
        output_tokens = self.client.messages.count_tokens(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": response
            }],
        )
        self.output_tokens.append(output_tokens.input_tokens)

    def print_total_tokens_usage(self) -> None:
        self.logger.info(f"😚💦💬 Total Prompt Tokens Usage: {sum(self.prompt_tokens)}")
        self.logger.info(f"💰 Total Prompt Tokens Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)):.3f}$")
        self.logger.info(f"🤖💬 Total Completion Tokens Usage: {sum(self.output_tokens)}")
        self.logger.info(f"💰 Total Completion Tokens Cost: {self._calculate_output_tokens_cost(sum(self.output_tokens)):.3f}$")
        self.logger.critical(f"🤡🤡🤡 Total Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)) + self._calculate_output_tokens_cost(sum(self.output_tokens)):.3f}$")
    