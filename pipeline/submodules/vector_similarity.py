import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple

def compute_vector_similarities(directions: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], float]:
    """
    Compute cosine similarity between refusal directions for all language pairs.
    
    Args:
        directions: Dictionary mapping language codes to refusal direction vectors
        
    Returns:
        Dictionary mapping language pairs to similarity scores
    """
    similarities = {}
    languages = list(directions.keys())
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i <= j:  # Only compute each pair once (including self-similarity)
                vec1 = directions[lang1]
                vec2 = directions[lang2]
                
                # Normalize vectors
                vec1_norm = vec1 / torch.norm(vec1)
                vec2_norm = vec2 / torch.norm(vec2)
                
                # Compute cosine similarity
                sim = torch.dot(vec1_norm, vec2_norm).item()
                similarities[(lang1, lang2)] = sim
                similarities[(lang2, lang1)] = sim  # Symmetry
    
    return similarities

def plot_similarity_heatmap(similarities: Dict[Tuple[str, str], float], 
                           languages: List[str], 
                           output_path: str):
    """
    Generate a heatmap visualization of similarity between refusal directions.
    
    Args:
        similarities: Dictionary mapping language pairs to similarity scores
        languages: List of language codes
        output_path: Path to save the heatmap
    """
    # Create similarity matrix
    n = len(languages)
    sim_matrix = np.zeros((n, n))
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            sim_matrix[i, j] = similarities.get((lang1, lang2), 0)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis",
               xticklabels=languages, yticklabels=languages)
    plt.title("Cosine Similarity Between Refusal Directions")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Save raw data
    with open(output_path.replace('.png', '.json'), 'w') as f:
        json_data = {f"{l1}_{l2}": similarities[(l1, l2)] for l1 in languages for l2 in languages}
        json.dump(json_data, f, indent=4)

def plot_cross_lingual_effectiveness(effectiveness: Dict[Tuple[str, str], float],
                                    languages: List[str],
                                    output_path: str,
                                    title: str):
    """
    Generate a heatmap visualization of cross-lingual effectiveness.
    
    Args:
        effectiveness: Dictionary mapping (source_lang, target_lang) to effectiveness scores
        languages: List of language codes
        output_path: Path to save the heatmap
        title: Title for the heatmap
    """
    # Create effectiveness matrix
    n = len(languages)
    eff_matrix = np.zeros((n, n))
    
    for i, src_lang in enumerate(languages):
        for j, tgt_lang in enumerate(languages):
            eff_matrix[i, j] = effectiveness.get((src_lang, tgt_lang), 0)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(eff_matrix, annot=True, fmt=".2f", cmap="coolwarm",
               xticklabels=languages, yticklabels=languages)
    plt.title(title)
    plt.xlabel("Target Language (Prompt Language)")
    plt.ylabel("Source Language (Direction Vector)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Save raw data
    with open(output_path.replace('.png', '.json'), 'w') as f:
        json_data = {f"{l1}_{l2}": effectiveness[(l1, l2)] for l1 in languages for l2 in languages}
        json.dump(json_data, f, indent=4) 