from utils.utils import load_prompts_from_dir_into_dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from typing import Tuple
import torch
from transformers import BitsAndBytesConfig


MODELS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Original",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

TEMPLATES = {
    "llama2": "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
    "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "llama4": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "tinyllama": "<|system|>\n{}<|user|>\n{}<|assistant|>",
}

QUANTIZATION_CONFIG_4_BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

QUANTIZATION_CONFIG_8_BIT = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
)


class Config:
    def __init__(
        self,
        model_name,
        device,
        parallel=False,
        max_new_tokens=None,
        prompts_folder=None,
    ):
        quantization = None
        if model_name in ["llama3_8bit", "llama3_4bit"]:
            quantization = model_name.split("_")[-1]
            model_name = "llama3"

        self.model_name = model_name
        self.model_config = MODELS[model_name]
        self.chat_template = TEMPLATES[model_name]
        self.device = device
        self.parallel = parallel
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.prompts = None

        if model_name in MODELS:
            if quantization:
                model, tokenizer = load_model_with_quantization(self, quantization)
            else:
                model, tokenizer = load_model(self)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.model = model
        self.tokenizer = tokenizer

        prompts = load_prompts_from_dir_into_dict(prompts_folder)
        self.prompts = prompts


def load_model_with_quantization(
    config: Config, quantization: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if quantization == "4bit":
        quantization_config = QUANTIZATION_CONFIG_4_BIT
    elif quantization == "8bit":
        quantization_config = QUANTIZATION_CONFIG_8_BIT
    else:
        raise ValueError(f"Invalid quantization config: {quantization}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_config,
        quantization_config=quantization_config,
        device_map="auto" if config.parallel else config.device,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model(config: Config) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_config,
        device_map="auto" if config.parallel else config.device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_config)
    return model, tokenizer


def generate(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    config: dict,
) -> str:
    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=config.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
    )
