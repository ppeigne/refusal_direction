import torch
import random
import json
import os
import argparse
import gc
from torch.cuda.amp import autocast

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from pipeline.utils.translation_utils import (
    translate_dataset, translate_instructions,
    ALL_LANGS, HIGH_RESOURCE_LANGS, MEDIUM_RESOURCE_LANGS, LOW_RESOURCE_LANGS
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.vector_similarity import (
    compute_vector_similarities, 
    plot_similarity_heatmap,
    plot_cross_lingual_effectiveness
)

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, batch_size=8):
    """
    Filter datasets based on refusal scores with Qwen-specific memory optimizations.
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    def get_scores_in_batches(model, dataset, tokenize_fn, refusal_toks):
        all_scores = []
        
        # Use smaller batch size and mixed precision for Qwen2
        with torch.cuda.amp.autocast():  # Enable mixed precision
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                try:
                    scores = get_refusal_scores(model, batch, tokenize_fn, refusal_toks, batch_size=batch_size)
                    all_scores.extend(scores.tolist())
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        # If OOM occurs, try with even smaller batch
                        torch.cuda.empty_cache()
                        smaller_batch = batch[:len(batch)//2]
                        scores = get_refusal_scores(model, smaller_batch, tokenize_fn, refusal_toks, batch_size=batch_size//2)
                        all_scores.extend(scores.tolist())
                        # Process remaining half
                        scores = get_refusal_scores(model, batch[len(batch)//2:], tokenize_fn, refusal_toks, batch_size=batch_size//2)
                        all_scores.extend(scores.tolist())
                    else:
                        raise e
                
                # Clear memory between batches
                torch.cuda.empty_cache()
            
        return torch.tensor(all_scores, device=model.device)

    if cfg.filter_train:
        harmful_train_scores = get_scores_in_batches(model_base.model, harmful_train, 
                                                   model_base.tokenize_instructions_fn, model_base.refusal_toks)
        torch.cuda.empty_cache()  # Clear between datasets
        
        harmless_train_scores = get_scores_in_batches(model_base.model, harmless_train,
                                                    model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_scores_in_batches(model_base.model, harmful_val,
                                                 model_base.tokenize_instructions_fn, model_base.refusal_toks)
        torch.cuda.empty_cache()  # Clear between datasets
        
        harmless_val_scores = get_scores_in_batches(model_base.model, harmless_val,
                                                  model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmful_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, language=None):
    """Generate and save candidate directions."""
    artifact_dir = cfg.language_artifact_path(language) if language else cfg.artifact_path()
    artifact_subdir = os.path.join(artifact_dir, 'generate_directions')
    
    if not os.path.exists(artifact_subdir):
        os.makedirs(artifact_subdir)

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_subdir)

    torch.save(mean_diffs, os.path.join(artifact_subdir, 'mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, language=None):
    """Select and save the direction."""
    artifact_dir = cfg.language_artifact_path(language) if language else cfg.artifact_path()
    artifact_subdir = os.path.join(artifact_dir, 'select_direction')
    
    if not os.path.exists(artifact_subdir):
        os.makedirs(artifact_subdir)

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=artifact_subdir
    )

    with open(f'{artifact_dir}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer, "language": language or "en"}, f, indent=4)

    torch.save(direction, f'{artifact_dir}/direction.pt')

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, language=None):
    """Generate and save completions for a dataset."""
    artifact_dir = cfg.language_artifact_path(language) if language else cfg.artifact_path()
    completions_dir = os.path.join(artifact_dir, 'completions')
    
    if not os.path.exists(completions_dir):
        os.makedirs(completions_dir)

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    lang_suffix = f"_{language}" if language else ""
    with open(f'{completions_dir}/{dataset_name}_{intervention_label}{lang_suffix}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)
    
    return completions

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies, language=None):
    """Evaluate completions and save results for a dataset."""
    artifact_dir = cfg.language_artifact_path(language) if language else cfg.artifact_path()
    completions_dir = os.path.join(artifact_dir, 'completions')
    
    lang_suffix = f"_{language}" if language else ""
    completions_file = os.path.join(completions_dir, f"{dataset_name}_{intervention_label}{lang_suffix}_completions.json")
    
    with open(completions_file, 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(completions_dir, f"{dataset_name}_{intervention_label}{lang_suffix}_evaluations.json"),
    )

    with open(os.path.join(completions_dir, f"{dataset_name}_{intervention_label}{lang_suffix}_evaluations.json"), "w") as f:
        json.dump(evaluation, f, indent=4)
    
    return evaluation

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, language=None, batch_size=None):
    """Evaluate loss on datasets."""
    artifact_dir = cfg.language_artifact_path(language) if language else cfg.artifact_path()
    loss_dir = os.path.join(artifact_dir, 'loss_evals')
    
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    lang_suffix = f"_{language}" if language else ""
    completions_file_path = os.path.join(artifact_dir, f'completions/harmless_baseline{lang_suffix}_completions.json')

    if batch_size is None:
        batch_size = cfg.ce_loss_batch_size

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=completions_file_path)

    with open(f'{loss_dir}/{intervention_label}{lang_suffix}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)
    
    return loss_evals

def clear_gpu_memory():
    """Force clear CUDA memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
def get_language_batch_size(language, base_batch_size):
    """
    Adjust batch size based on language to account for token differences.
    
    Args:
        language: Language code
        base_batch_size: Base batch size for English
        
    Returns:
        Adjusted batch size for the language
    """
    # Languages that typically use more tokens per text
    if language in ["th", "am", "zh", "ru"]:
        return max(1, base_batch_size // 2)  # Reduce batch size by half for these languages
    return base_batch_size

def run_monolingual_pipeline(cfg, model_base, language=None):
    """Run pipeline for a single language with memory optimizations."""
    print(f"\n{'='*20} Running pipeline for {language or 'English'} {'='*20}\n")
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Create language-specific artifact directory
    if language:
        lang_dir = cfg.language_artifact_path(language)
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)
    
    # Adjust batch sizes based on language
    batch_size = get_language_batch_size(language, cfg.ce_loss_batch_size)
    print(f"Using batch size {batch_size} for {language or 'English'}")
    
    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    
    # Translate datasets if needed
    if language and cfg.translate_datasets:
        harmful_train = translate_instructions(harmful_train, language)
        harmless_train = translate_instructions(harmless_train, language)
        harmful_val = translate_instructions(harmful_val, language)
        harmless_val = translate_instructions(harmless_val, language)
    
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )
    clear_gpu_memory()

    # 1. Generate candidate refusal directions
    try:
        with autocast():  # Use mixed precision to reduce memory usage
            candidate_directions = generate_and_save_candidate_directions(
                cfg, model_base, harmful_train, harmless_train, language
            )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM error during direction generation for {language}. Reducing dataset size and retrying...")
            # Reduce dataset size and retry
            harmful_train = harmful_train[:len(harmful_train)//2]
            harmless_train = harmless_train[:len(harmless_train)//2]
            clear_gpu_memory()
            candidate_directions = generate_and_save_candidate_directions(
                cfg, model_base, harmful_train, harmless_train, language
            )
        else:
            raise e
    
    clear_gpu_memory()
    
    # 2. Select the most effective refusal direction
    try:
        with autocast():
            pos, layer, direction = select_and_save_direction(
                cfg, model_base, harmful_val, harmless_val, candidate_directions, language
            )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM error during direction selection for {language}. Reducing dataset size and retrying...")
            # Reduce dataset size and retry
            harmful_val = harmful_val[:len(harmful_val)//2]
            harmless_val = harmless_val[:len(harmless_val)//2]
            clear_gpu_memory()
            pos, layer, direction = select_and_save_direction(
                cfg, model_base, harmful_val, harmless_val, candidate_directions, language
            )
        else:
            raise e
    
    clear_gpu_memory()

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        clear_gpu_memory()
        
        dataset = load_dataset(dataset_name)
        if language and cfg.translate_datasets:
            dataset = translate_dataset(dataset, language)
        
        # Use smaller subset if needed for low-resource languages
        if language in ["th", "am", "sw", "zh"] and len(dataset) > 50:
            dataset = dataset[:50]  # Use only first 50 examples for memory-intensive languages
            
        with autocast():
            generate_and_save_completions_for_dataset(
                cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 
                'baseline', dataset_name, dataset, language
            )
            clear_gpu_memory()
            
            generate_and_save_completions_for_dataset(
                cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 
                'ablation', dataset_name, dataset, language
            )
            clear_gpu_memory()
            
            generate_and_save_completions_for_dataset(
                cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 
                'actadd', dataset_name, dataset, language
            )
            clear_gpu_memory()

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(
            cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, language=language
        )
        evaluate_completions_and_save_results_for_dataset(
            cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, language=language
        )
        evaluate_completions_and_save_results_for_dataset(
            cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, language=language
        )
    
    # 4a. Generate and save completions on harmless evaluation dataset
    clear_gpu_memory()
    
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
    if language and cfg.translate_datasets:
        harmless_test = translate_instructions(harmless_test, language)
    
    # Adjust test set size for memory-intensive languages
    if language in ["th", "am", "sw", "zh"] and len(harmless_test) > 50:
        harmless_test = harmless_test[:50]
        
    with autocast():
        generate_and_save_completions_for_dataset(
            cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 
            'baseline', 'harmless', dataset=harmless_test, language=language
        )
        clear_gpu_memory()
    
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    
    with autocast():
        generate_and_save_completions_for_dataset(
            cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 
            'actadd', 'harmless', dataset=harmless_test, language=language
        )
        clear_gpu_memory()

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(
        cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies, language=language
    )
    evaluate_completions_and_save_results_for_dataset(
        cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies, language=language
    )

    # 5. Evaluate loss on harmless datasets with adjusted batch size for language
    with autocast():
        evaluate_loss_for_datasets(
            cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', language, 
            batch_size=batch_size
        )
        clear_gpu_memory()
        
        evaluate_loss_for_datasets(
            cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', language,
            batch_size=batch_size
        )
        clear_gpu_memory()
        
        evaluate_loss_for_datasets(
            cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', language,
            batch_size=batch_size
        )
        clear_gpu_memory()
    
    return pos, layer, direction

def run_cross_lingual_experiments(cfg, model_base, language_directions):
    """Run cross-lingual experiments with memory optimizations."""
    print("\n" + "="*20 + " Running cross-lingual experiments " + "="*20 + "\n")
    
    # Create cross-lingual artifact directory
    cross_lingual_dir = os.path.join(cfg.artifact_path(), "cross_lingual")
    if not os.path.exists(cross_lingual_dir):
        os.makedirs(cross_lingual_dir)
    
    # Load harmless and harmful test datasets
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
    harmful_test = random.sample(load_dataset_split(harmtype='harmful', split='test'), cfg.n_test)
    
    # Reduce test set size to manage memory
    harmless_test = harmless_test[:50]
    harmful_test = harmful_test[:50]
    
    languages = list(language_directions.keys())
    
    # Store cross-lingual effectiveness results
    ablation_effectiveness = {}  # (source_lang, target_lang) -> effectiveness score
    addition_effectiveness = {}  # (source_lang, target_lang) -> effectiveness score
    
    # For each language pair, apply source language direction to target language prompts
    for source_lang in languages:
        for target_lang in languages:
            clear_gpu_memory()
            print(f"Testing {source_lang} direction on {target_lang} prompts...")
            
            source_pos, source_layer, source_direction = language_directions[source_lang]
            
            # Translate test datasets to target language
            if target_lang != "en":
                target_harmful_test = translate_instructions(harmful_test, target_lang)
                target_harmless_test = translate_instructions(harmless_test, target_lang)
            else:
                target_harmful_test = harmful_test
                target_harmless_test = harmless_test
            
            # Setup hooks with source language direction
            ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, source_direction)
            actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[source_layer], 
                                                     get_activation_addition_input_pre_hook(vector=source_direction, coeff=+1.0))], []
            
            # Generate completions with ablation (for harmful prompts)
            try:
                with autocast():
                    cross_completions = generate_and_save_completions_for_dataset(
                        cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks,
                        f'cross_ablation_{source_lang}', 'harmful_test', 
                        dataset=target_harmful_test, language=target_lang
                    )
                    clear_gpu_memory()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM error during cross-lingual ablation for {source_lang}->{target_lang}. Reducing dataset size.")
                    # Try with a smaller set
                    clear_gpu_memory()
                    cross_completions = generate_and_save_completions_for_dataset(
                        cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks,
                        f'cross_ablation_{source_lang}', 'harmful_test', 
                        dataset=target_harmful_test[:25], language=target_lang
                    )
                else:
                    raise e
            
            # Evaluate cross-lingual ablation
            cross_eval = evaluate_completions_and_save_results_for_dataset(
                cfg, f'cross_ablation_{source_lang}', 'harmful_test',
                eval_methodologies=cfg.jailbreak_eval_methodologies, language=target_lang
            )
            
            # Store effectiveness (jailbreak success rate)
            if "substring_matching_success_rate" in cross_eval:
                ablation_effectiveness[(source_lang, target_lang)] = cross_eval["substring_matching_success_rate"]
            
            clear_gpu_memory()
            
            # Generate completions with addition (for harmless prompts)
            try:
                with autocast():
                    cross_completions = generate_and_save_completions_for_dataset(
                        cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks,
                        f'cross_actadd_{source_lang}', 'harmless_test', 
                        dataset=target_harmless_test, language=target_lang
                    )
                    clear_gpu_memory()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM error during cross-lingual addition for {source_lang}->{target_lang}. Reducing dataset size.")
                    # Try with a smaller set
                    clear_gpu_memory()
                    cross_completions = generate_and_save_completions_for_dataset(
                        cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks,
                        f'cross_actadd_{source_lang}', 'harmless_test', 
                        dataset=target_harmless_test[:25], language=target_lang
                    )
                else:
                    raise e
            
            # Evaluate cross-lingual addition
            cross_eval = evaluate_completions_and_save_results_for_dataset(
                cfg, f'cross_actadd_{source_lang}', 'harmless_test',
                eval_methodologies=cfg.refusal_eval_methodologies, language=target_lang
            )
            
            # Store effectiveness (refusal rate)
            if "substring_matching_success_rate" in cross_eval:
                addition_effectiveness[(source_lang, target_lang)] = cross_eval["substring_matching_success_rate"]
            
            clear_gpu_memory()
    
    # Generate and save cross-lingual effectiveness heatmaps
    plot_cross_lingual_effectiveness(
        ablation_effectiveness, languages,
        os.path.join(cross_lingual_dir, "ablation_effectiveness.png"),
        "Cross-lingual Refusal Vector Ablation Effectiveness (Higher = Better Jailbreak)"
    )
    
    plot_cross_lingual_effectiveness(
        addition_effectiveness, languages,
        os.path.join(cross_lingual_dir, "addition_effectiveness.png"),
        "Cross-lingual Refusal Vector Addition Effectiveness (Higher = Better Refusal)"
    )
    
    return ablation_effectiveness, addition_effectiveness

def analyze_vector_similarities(cfg, language_directions):
    """Analyze similarities between refusal vectors across languages."""
    print("\n" + "="*20 + " Analyzing vector similarities " + "="*20 + "\n")
    
    # Create directory for similarity analysis
    similarity_dir = os.path.join(cfg.artifact_path(), "vector_similarity")
    if not os.path.exists(similarity_dir):
        os.makedirs(similarity_dir)
    
    # Extract only direction vectors from language_directions
    directions = {lang: direction for lang, (_, _, direction) in language_directions.items()}
    
    # Compute similarities between all direction vectors
    similarities = compute_vector_similarities(directions)
    
    # Plot and save similarity heatmap
    plot_similarity_heatmap(
        similarities, 
        list(directions.keys()),
        os.path.join(similarity_dir, "direction_similarities.png")
    )
    
    return similarities

def run_pipeline(model_path):
    """Run the full pipeline with memory optimizations."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    # Set languages or use defaults
    if not cfg.languages:
        cfg.languages = ALL_LANGS
    
    model_base = construct_model_base(cfg.model_path)
    
    # Dictionary to store directions for each language
    language_directions = {}
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Run monolingual pipeline for English (baseline)
    en_pos, en_layer, en_direction = run_monolingual_pipeline(cfg, model_base)
    language_directions["en"] = (en_pos, en_layer, en_direction)
    
    # Run monolingual pipeline for each additional language
    for lang in cfg.languages:
        if lang != "en":  # English already processed
            clear_gpu_memory()  # Clear memory between languages
            pos, layer, direction = run_monolingual_pipeline(cfg, model_base, lang)
            language_directions[lang] = (pos, layer, direction)
    
    # Run cross-lingual experiments if enabled
    if cfg.cross_lingual_analysis:
        clear_gpu_memory()
        ablation_effectiveness, addition_effectiveness = run_cross_lingual_experiments(
            cfg, model_base, language_directions
        )
    
    # Analyze vector similarities if enabled
    if cfg.similarity_analysis:
        clear_gpu_memory()
        similarities = analyze_vector_similarities(cfg, language_directions)
    
    # Save a summary of findings
    summary = {
        "model": model_alias,
        "languages": cfg.languages,
        "direction_positions": {lang: {"pos": pos, "layer": layer} 
                               for lang, (pos, layer, _) in language_directions.items()},
    }
    
    with open(os.path.join(cfg.artifact_path(), "multilingual_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*20 + " Pipeline complete " + "="*20 + "\n")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
