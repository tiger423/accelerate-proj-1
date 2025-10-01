Need a HF key to run

  Usage:

  python memory_predictor.py --model <model_name> [--token  <hf_token>]

  Examples:

  # Public models (no token needed)
  python memory_predictor.py --model gpt2
  
  python memory_predictor.py --model google/flan-t5-base

  # Meta/Facebook models (require HF token)
  python memory_predictor.py --model facebook/opt-30b --token YOUR_TOKEN
  
  python memory_predictor.py --model meta-llama/Llama-2-7b-hf --token YOUR_TOKEN

  Features:

  ✓ Model-agnostic - Works with any HuggingFace model
  ✓ No downloading - Fetches only config metadata (instant)
  ✓ Hardware detection - Analyzes your GPU/RAM/disk automatically
  ✓ Multiple strategies - Provides different allocation options (CPU-only, GPU+CPU, etc.)
  ✓ Feasibility checking - Tells you if your hardware can run the model
  ✓ Speed estimates - Predicts performance for each strategy
  ✓ JSON export - Saves results to memory_analysis.json
