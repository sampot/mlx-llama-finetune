#!/bin/bash

set -e

model_name="llama3.1-spk1"
source_model="meta-llama/Meta-Llama-3.1-8B"
local_model="./models/mlx"
fused_model="./models/llama-3.1-fused"

function data() {
  if [ ! -d "./data" ] || [ ! -f "./data/train.jsonl" ] || [ ! -f "./data/valid.jsonl" ]; then
    echo "Download and prepare datesets."
    poetry run python prepare_data.py
  fi
  echo "Datasets ready."
}

function fetch() {
  if [ ! -d "./models/mlx" ]; then
    echo "Fetch and quantize model."
    poetry run mlx_lm.convert \
      --hf-path "${source_model}" \
      --mlx-path "${local_model}" \
      --quantize \
      --q-bits 4
    echo "Model fetched."
  else
    echo "Model exist."
  fi
}

function train() {
  if [ ! -d "./adapters" ] || [ ! -f "./adapters/adapter_config.json" ]; then
    data
    fetch
    echo "Training with config file lora_config.yaml"
    poetry run mlx_lm.lora --config lora_config.yaml --model ${local_model}
  fi
  echo "Model trained with Lora."
}

function test() {
  if [ ! -f "./data/test.jsonl" ]; then
    echo "Test dataset doesn't exist."
    return
  fi

  if [ ! -f "./adapters/adapter_config.json" ]; then
    echo "Model not trained yet."
    return
  fi

  echo "Test model."
  poetry run mlx_lm.lora \
    --model "${local_model}" \
    --adapter-path ./adapters \
    --data ./data \
    --test
}

function fuse() {
  if [ ! -d "${fused_model}" ]; then
    train
    echo "Fuse fine-tuned weights with original model"

    poetry run mlx_lm.fuse \
      --model ${local_model} \
      --save-path "${fused_model}" \
      --adapter-path ./adapters \
      --de-quantize
  fi
  echo "Model fused."
}

function convert() {
  if [ ! -f "./modesl/model.gguf" ]; then
    fuse
    echo "Converting weights to GGUF format"
    poetry run python ./llama.cpp/convert_hf_to_gguf.py \
      --outfile ./models/model.gguf \
      --outtype q8_0 \
      --model-name ${model_name} \
      ${fused_model}
  fi
  echo "Model converted to GGUF."
}

function create() {
  if $(ollama list | grep -q "${model_name}"); then
    echo "Model ${model_name} exist."
    return
  else
    convert
    echo "Create Ollama model for inference locally."
    ollama create -f Modelfile ${model_name}
    echo "Model created for Ollama."
  fi
}

function run() {
  echo "Run fine-tuned model with Ollama."
  create
  ollama run ${model_name}
}

function clean() {
  echo "Cleaning up"
  rm -rf ./models
  rm -rf ./adapters
  rm -rf ./data
}

function help() {
  echo "$0 <command>"
}

case "$1" in
data)
  data
  ;;
fetch)
  fetch
  ;;
train)
  train
  ;;
test)
  test
  ;;
fuse)
  fuse
  ;;
convert)
  convert
  ;;
create)
  create
  ;;
run)
  run
  ;;
clean)
  clean
  ;;
*)
  help
  ;;
esac
