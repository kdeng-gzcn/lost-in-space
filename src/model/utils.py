from src.model import *

def load_model(model_id):
    model_mapping = {
        # LLM
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaInstruct,
        "meta-llama/Llama-3.1-8B-Instruct": LlamaInstruct,
        "Qwen/Qwen2.5-7B-Instruct": QwenInstruct,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": QwenInstruct,
        "gpt-4": GPTInstruct,
        "gpt-4o-text-only": GPTInstruct,

        # VLM
        "microsoft/Phi-3.5-vision-instruct": Phi3VisionInstruct, # ❌
        "microsoft/Phi-4-multimodal-instruct": Phi4VisionInstruct, # ❌

        "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321": VLMR1VisionInstruct,
        "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps": VLMR1VisionInstruct,
        "zai-org/GLM-4.1V-9B-Thinking": GLMVisionInstruct,
        "THUDM/GLM-4.1V-9B-Thinking": GLMVisionInstruct, # ✅

        "remyxai/SpaceQwen2.5-VL-3B-Instruct": QwenVisionInstruct, # ✅

        "Qwen/Qwen2.5-VL-3B-Instruct": QwenVisionInstruct, # ✅
        "Qwen/Qwen2.5-VL-7B-Instruct": QwenVisionInstruct, # ✅
        "Qwen/Qwen2.5-VL-32B-Instruct": QwenVisionInstruct, # ✅
        "Qwen/Qwen2.5-VL-72B-Instruct": QwenVisionInstruct, # ✅

        "Qwen/Qwen3-VL-30B-A3B-Instruct": Qwen3VisionInstruct,
        "Qwen/Qwen3-VL-4B-Instruct": Qwen3VisionInstruct, # ✅
        "Qwen/Qwen3-VL-8B-Instruct": Qwen3VisionInstruct, # ✅
        "Qwen/Qwen3-VL-32B-Instruct": Qwen3VisionInstruct, # ✅
        
        "Qwen/Qwen3-VL-4B-Thinking": Qwen3VisionInstruct,
        "Qwen/Qwen3-VL-8B-Thinking": Qwen3VisionInstruct, # ✅

        "meta-llama/Llama-4-Scout-17B-16E-Instruct": Llama4VisionInstruct,

        "gpt-4o-mini": GPTVisionInstruct, # ✅
        "gpt-4o": GPTVisionInstruct, # ✅
        "gpt-4-turbo": GPTVisionInstruct, # ✅
        "gpt-5": GPT5, #

        "claude-sonnet-4-20250514": AnthropicVisionInstruct,

        "google/gemma-3-12b-it": Gemma3VisionInstruct, # ❌ TODO: triton

        "HuggingFaceM4/Idefics3-8B-Llama3": Idefics3VisionInstruct, # ✅
        # "deepseek-ai/deepseek-vl2-small": DeepseekVisionInstruct(model_id=model_id), # ❌ TODO: xformers
        "llava-hf/llama3-llava-next-8b-hf": LlavaNextVisionInstruct, # ✅
        "llava-hf/llava-onevision-qwen2-7b-ov-hf": LlavaOneVisionInstruct, # ✅
        "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": LlavaOneVisionInstruct,

        # "Salesforce/blip2-opt-2.7b": BLIP(model_id=model_name),
        "Salesforce/instructblip-vicuna-7b": BLIPVisionInstruct, # ✅
    }

    if model_id not in model_mapping:
        raise NotImplementedError(f"Model {model_id} not supported.")

    return model_mapping[model_id](model_id=model_id)
