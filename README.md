# Fine-tune Llama model with MLX

## Pre-requisites

1. You need a Hugging Face account and agreed to the terms for [Meta-Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

## Checkout code

```bash
git clone https://github.com/sampot/mlx-llama-finetune
```

## Install dependencies

```
poetry Install
pip install -r llama.cpp/requirements.txt
```

## Prepare training dataset

This is a time-consuming process. In this case, the dataset from mlx-examples is used instead.

## Convert to GGUF format

As Ollama can only use GGUF-formated model for inference, so we need to convert the model to GGUF with the script from llama.cpp project.

Before executing the convertion script, we need to ensure the llama.cpp version is compatible with the one in Ollama.
Otherewise, Ollama might not be able to create the model due to error such as `expecting tensor layers 292, got 291 instead`

The tag branch `b3418` is used.

```
cd llama.cpp
git checkout b3418
```

```
python ./llama.cpp/convert-hf-to-gguf.py --outfile ./models/model.gguf --outtype q8_0 ./models/llama3.1-spk1
```

## Run the fine-tuned model locally with Ollama

```bash
ollama run llama3.1-spk1
```
