import torch
import functools
from typing import List, Dict, Callable, Any, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Qwen chat templates are based on
# - Official examples from Qwen repo: https://github.com/QwenLM/Qwen/blob/5aa84bdfd3237b37f01bc88cd49b3279b9a71d0b/examples/vllm_wrapper.py#L32
# - Online guidelines: https://github.com/Ki-Seki/chat_prompt_templates?tab=readme-ov-file#qwen-prompt-template

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T

def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer-1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer-1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer-1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class QwenModel(ModelBase):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)

        refusal_keywords = [
            "I'm sorry",
            "I apologize",
            "I cannot",
            "I can't",
            "I am not able",
            "as an AI",
            "as a language model",
            "cannot fulfill",
            "cannot provide",
            "unable to",
            "not appropriate",
            "refuse to",
            "would not be",
            "ethical",
            "illegal"
        ]
        self.refusal_toks = self._get_all_refusal_tok_ids(refusal_keywords, self.tokenizer)

    def _get_all_refusal_tok_ids(self, refusal_keywords: List[str], tokenizer) -> List[int]:
        """Get all refusal token ids for the given tokenizer."""
        refusal_toks = []
        for keyword in refusal_keywords:
            keyword_toks = tokenizer.encode(keyword, add_special_tokens=False)
            refusal_toks.extend(keyword_toks)
        return refusal_toks

    def tokenize_instructions_fn(self, instructions: List[str], outputs: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize instructions for Qwen models.
        """
        if outputs is not None:
            messages = []
            for instruction, output in zip(instructions, outputs):
                message = [
                    {"role": "user", "content": instruction.strip()},
                    {"role": "assistant", "content": output.strip()}
                ]
                messages.append(message)
            
            encodings = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True
            )
        else:
            messages = []
            for instruction in instructions:
                message = [
                    {"role": "user", "content": instruction.strip()}
                ]
                messages.append(message)
            
            encodings = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True
            )
        
        return encodings

    def _load_model(self, model_name_or_path: str) -> PreTrainedModel:
        """
        Load model from HuggingFace.
        """
        # Check if this is a Qwen2 model (needs different parameters)
        is_qwen2 = "qwen2" in model_name_or_path.lower()
        
        # Set appropriate loading parameters based on model type
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        
        # Only add flash_attn for Qwen1 models that support it, not for Qwen2
        if not is_qwen2:
            model_kwargs["use_flash_attn"] = True
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        model.eval()
        return model

    def _load_tokenizer(self, model_name_or_path: str) -> 'AutoTokenizer':
        """
        Load tokenizer from HuggingFace.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return self.tokenize_instructions_fn

    def _get_eoi_toks(self):
        """
        Return the tokens which indicate end-of-instruction.
        For Qwen models, this is the assistant start tokens.
        """
        # Try to identify assistant tokens by looking at chat template
        # This is a heuristic approach since different Qwen versions might have different tokens
        if "qwen2" in self.model_name_or_path.lower():
            # For Qwen2 models
            return self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        else:
            # For Qwen1 models
            return self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        if "qwen2" in self.model_name_or_path.lower():
            return self.model.model.layers
        else:
            return self.model.transformer.h

    def _get_attn_modules(self):
        modules = []
        for block in self.model_block_modules:
            if "qwen2" in self.model_name_or_path.lower():
                modules.append(block.self_attn)
            else:
                modules.append(block.attn)
        return torch.nn.ModuleList(modules)

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        if "qwen2" in self.model_name_or_path.lower():
            # Implementation for Qwen2 would go here
            raise NotImplementedError("Orthogonalization for Qwen2 models not yet implemented")
        else:
            return functools.partial(orthogonalize_qwen_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        if "qwen2" in self.model_name_or_path.lower():
            # Implementation for Qwen2 would go here
            raise NotImplementedError("Activation addition for Qwen2 models not yet implemented")
        else:
            return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)