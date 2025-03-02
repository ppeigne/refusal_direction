# Refusal in Language Models Across Languages

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains code and results accompanying the paper "Refusal in Language Models Is Mediated by a Single Direction", with additional experiments investigating the cross-lingual properties of refusal directions.

- [Original Paper](https://arxiv.org/abs/2406.11717)
- [Blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## Setup

```bash
git clone https://github.com/andyrdt/refusal_direction.git
cd refusal_direction
source setup.sh
```

The setup script will prompt you for:
- A HuggingFace token (required to access gated models)
- A Together AI token (required to access the Together AI API, which is used for evaluating jailbreak safety scores)
- Google Cloud credentials (required for translation - set GOOGLE_APPLICATION_CREDENTIALS environment variable)

It will then set up a virtual environment and install the required packages.

### Setting up Google Cloud Translate API

For multilingual experiments, you need to:

1. Create a Google Cloud account and project
2. Enable the Cloud Translate API
3. Create a service account key and download it as JSON
4. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-google-credentials.json
```

## Translation Experiments

We extend the original work to investigate how refusal directions operate across languages. The experiments include:

1. Extracting refusal directions for multiple languages:
   - High-resource: English, French, Spanish
   - Medium-resource: German, Russian, Chinese
   - Low-resource: Thai, Swahili, Amharic

2. Cross-lingual analysis:
   - Testing how refusal vectors from one language perform on prompts in other languages
   - Measuring similarity between refusal vectors across languages

3. Visualizations:
   - Heatmaps of cross-lingual effectiveness
   - Similarity matrices between refusal directions

## Reproducing main results

To reproduce the main results from the paper, run the following command:

```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```
where `{model_path}` is the path to a HuggingFace model. For example, for Llama-3 8B Instruct, the model path would be `meta-llama/Meta-Llama-3-8B-Instruct`.

For Qwen models, you can run:
```bash
python3 -m pipeline.run_pipeline --model_path Qwen/Qwen-7B-Chat  # Qwen1 model
python3 -m pipeline.run_pipeline --model_path Qwen/Qwen2-7B-Instruct  # Qwen2 model
```

The pipeline performs the following steps:
1. Extract candiate refusal directions across languages
2. Select the most effective refusal direction for each language
3. Generate completions over harmful and harmless prompts, and evaluate refusal metrics for each language
4. Perform cross-lingual analysis by applying refusal vectors from one language to prompts in other languages
5. Analyze similarities between refusal vectors across languages

Artifacts will be stored in language-specific directories:
- `pipeline/runs/{model_alias}/lang_{language_code}/`

Cross-lingual analysis results are stored in:
- `pipeline/runs/{model_alias}/cross_lingual/`

Vector similarity analysis results are stored in:
- `pipeline/runs/{model_alias}/vector_similarity/`

## Memory Optimization

The multilingual experiments may require significant GPU memory, especially for:
- Languages with non-Latin scripts (requiring more tokens per text)
- Larger models (7B+)

If you encounter OOM (Out of Memory) errors, consider:
1. Using fewer languages: `--languages en,fr,th`
2. Using a smaller model (Gemma-2B, Qwen-1.8B)
3. Reducing batch sizes in the config

## Minimal demo Colab

As part of our [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), we included a minimal demo of bypassing refusal. This demo is available as a [Colab notebook](https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw).

## As featured in

Since publishing our initial [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) in April 2024, our methodology has been independently reproduced and used many times. In particular, we acknowledge [Fail](https://huggingface.co/failspy)[Spy](https://x.com/failspy) for their work in reproducing and extending our methodology.

Our work has been featured in:
- [HackerNews](https://news.ycombinator.com/item?id=40242939)
- [Last Week in AI podcast](https://open.spotify.com/episode/2E3Fc50GVfPpBvJUmEwlOU)
- [Llama 3 hackathon](https://x.com/AlexReibman/status/1789895080754491686)
- [Applying refusal-vector ablation to a Llama 3 70B agent](https://www.lesswrong.com/posts/Lgq2DcuahKmLktDvC/applying-refusal-vector-ablation-to-a-llama-3-70b-agent)
- [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration)


## Citing this work

If you find this work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2406.11717):
```tex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Andy Arditi and Oscar Obeso and Aaquib Syed and Daniel Paleka and Nina Panickssery and Wes Gurnee and Neel Nanda},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```