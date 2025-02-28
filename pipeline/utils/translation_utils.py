import os
from typing import Dict, List, Tuple
from google.cloud import translate_v2 as translate
from tqdm import tqdm

# Language resource categories
HIGH_RESOURCE_LANGS = ["fr", "es"]  # French, Spanish
MEDIUM_RESOURCE_LANGS = ["de", "ru", "zh"]  # German, Russian, Chinese
LOW_RESOURCE_LANGS = ["th", "sw", "am"]  # Thai, Swahili, Amharic

ALL_LANGS = ["en"] + HIGH_RESOURCE_LANGS + MEDIUM_RESOURCE_LANGS + LOW_RESOURCE_LANGS

def get_translate_client():
    """Initialize Google Translate client."""
    assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), \
        "GOOGLE_APPLICATION_CREDENTIALS environment variable must be set"
    return translate.Client()

def translate_text(client, text: str, target_language: str) -> str:
    """Translate a text to target language."""
    if target_language == "en":
        return text
    
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]

def batch_translate(texts: List[str], target_language: str, batch_size: int = 10) -> List[str]:
    """Translate a batch of texts."""
    client = get_translate_client()
    translated_texts = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating to {target_language}"):
        batch = texts[i:i+batch_size]
        translated_batch = [translate_text(client, text, target_language) for text in batch]
        translated_texts.extend(translated_batch)
    
    return translated_texts

def translate_dataset(dataset: List[Dict], target_language: str) -> List[Dict]:
    """Translate prompts and keep the original structure."""
    prompts = [item["prompt"] for item in dataset]
    
    translated_prompts = batch_translate(prompts, target_language)
    
    translated_dataset = []
    for i, item in enumerate(dataset):
        translated_item = item.copy()
        translated_item["prompt"] = translated_prompts[i]
        translated_item["lang"] = target_language
        translated_dataset.append(translated_item)
    
    return translated_dataset

def translate_instructions(instructions: List[str], target_language: str) -> List[str]:
    """Translate a list of instructions."""
    return batch_translate(instructions, target_language) 