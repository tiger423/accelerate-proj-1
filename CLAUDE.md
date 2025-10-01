# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains experimental scripts for running large language models (specifically Facebook's OPT models) using HuggingFace Transformers and Accelerate libraries. The scripts demonstrate how to load and run massive models (30B-66B parameters) by distributing layers across GPU, CPU, and disk storage.

## Prerequisites

- HuggingFace account with API token
- Python 3.12 environment (py312)
- Required packages: `transformers`, `accelerate`, `torch`, `bitsandbytes` (optional)
- Significant disk space for model offloading (100GB+ recommended)
- GPU with VRAM (optional, but improves performance)

## Running the Scripts

All scripts require a HuggingFace API token. Set the `hf_token` variable in the script before running.

### Basic execution:
```bash
python accelerate_opt-secret-removed.py
```

### Script variants:

1. **accelerate_opt-secret-removed.py** - Base script for OPT-66B with GPU/CPU/disk distribution
2. **accelerate_opt-d-1-1-secret-removed.py** - OPT-30B with full CPU offload (`max_memory={0: "0GiB", "cpu":"20GiB"}`)
3. **accelerate_opt-d-1-2-secret-removed.py** - OPT-30B with CPU offload, inputs on CPU
4. **accelerate_opt-d-1-3-secret-removed.py** - OPT-30B with 5GB GPU allocation (`max_memory={0: "5GiB", "cpu":"20GiB"}`)
5. **accelerate_opt-d-1-4-secret-removed.py** - OPT-30B with 5GB GPU and inputs on GPU

## Architecture Details

### Memory Distribution Strategy

The scripts use HuggingFace Accelerate's `device_map="auto"` feature to automatically distribute model layers across available devices:

- **GPU (CUDA device 0)**: Embedding layers and initial decoder layers
- **CPU**: Middle decoder layers when GPU memory is insufficient
- **Disk (./swap directory)**: Remaining decoder layers are offloaded to disk

The `max_memory` parameter controls this distribution:
- `{0: "0GiB", "cpu":"20GiB"}` - Forces all layers to CPU/disk (no GPU usage)
- `{0: "5GiB", "cpu":"20GiB"}` - Allocates 5GB to GPU, rest to CPU/disk

### Model Loading Process

1. Model checkpoint shards are downloaded from HuggingFace Hub
2. Layers are loaded and distributed according to `device_map` and `max_memory`
3. `offload_folder='./swap'` specifies where disk-offloaded layers are stored
4. Final device map is printed via `model.hf_device_map`

### Inference Behavior

- Input tensors must be on the correct device: `.to(0)` for GPU or `.to('cpu')` for CPU
- When model is offloaded to CPU/disk, inputs should typically go to CPU
- Generation is much slower with disk offloading (~6-8 minutes for 30 tokens with OPT-30B, ~16 minutes with OPT-66B)

## Performance Characteristics

Based on output logs in the scripts:

- **OPT-66B** (263GB model): ~960 seconds for loading + generation with GPU/CPU/disk split
- **OPT-30B** (120GB model):
  - Full CPU offload: ~362-375 seconds
  - 5GB GPU allocation: ~500-509 seconds

## Supported Models

The scripts support various OPT model sizes:
- facebook/opt-125m
- facebook/opt-350m
- facebook/opt-1.3b
- facebook/opt-2.7b
- facebook/opt-6.7b
- facebook/opt-13b
- facebook/opt-30b
- facebook/opt-66b

Change the `my_model` variable to use different model sizes.

## Notes

- All Python files have "-secret-removed" suffix indicating HF tokens have been sanitized
- Scripts originated from Google Colab notebooks (see file headers)
- Quantization support exists but is commented out (BitsAndBytesConfig)
- The `use_fast=False` parameter for tokenizer is intentional for compatibility
