import os

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 100
    n_val: int = 32
    filter_train: bool = True
    filter_val: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    max_new_tokens: int = 512
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "llamaguard2")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048
    
    # Language settings
    languages: List[str] = None  # If None, defaults to all languages in translation_utils
    translate_datasets: bool = True  # Whether to translate datasets
    cross_lingual_analysis: bool = True  # Whether to perform cross-lingual analysis
    similarity_analysis: bool = True  # Whether to analyze similarities between directions

    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)
        
    def language_artifact_path(self, language: str) -> str:
        return os.path.join(self.artifact_path(), f"lang_{language}")