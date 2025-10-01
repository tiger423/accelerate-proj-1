"""
Memory Allocation Predictor for Large Language Models

This script analyzes your hardware and predicts optimal memory allocation strategies
for running large models with HuggingFace Transformers and Accelerate.

Usage:
    python memory_predictor.py --model facebook/opt-30b --token YOUR_HF_TOKEN
"""

import argparse
import json
import torch
from transformers import AutoConfig
from huggingface_hub import HfApi
import psutil
import shutil


def get_model_info(model_name, hf_token=None):
    """
    Fetch model configuration and calculate memory requirements.

    Returns:
        dict: Model metadata including size, layers, parameters
    """
    print(f"\n[INFO] Fetching model info for: {model_name}")

    try:
        # Load config without downloading model weights
        config = AutoConfig.from_pretrained(model_name, token=hf_token)

        # Calculate model size
        num_parameters = config.num_parameters if hasattr(config, 'num_parameters') else None

        # If num_parameters not in config, estimate from architecture
        if num_parameters is None:
            if hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                # Rough estimation for transformer models
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else 50000

                # Estimate: embedding + (layer_norm + attention + mlp) per layer
                params_per_layer = 12 * hidden_size * hidden_size  # Rough approximation
                embedding_params = vocab_size * hidden_size
                num_parameters = embedding_params + (num_layers * params_per_layer)

        # Get actual model size from HuggingFace Hub (more accurate)
        try:
            api = HfApi(token=hf_token)
            model_info = api.model_info(model_name)

            # Sum up all safetensors/bin file sizes
            total_bytes = 0
            if hasattr(model_info, 'siblings'):
                for file in model_info.siblings:
                    if file.rfilename.endswith(('.safetensors', '.bin')):
                        total_bytes += file.size if hasattr(file, 'size') else 0

            model_size_gb = total_bytes / (1024**3)
        except:
            # Fallback to parameter-based calculation
            # Assume float16 (2 bytes per parameter)
            model_size_gb = (num_parameters * 2) / (1024**3) if num_parameters else 0

        num_layers = getattr(config, 'num_hidden_layers',
                            getattr(config, 'n_layers',
                                   getattr(config, 'num_layers', None)))

        return {
            'model_name': model_name,
            'num_parameters': num_parameters,
            'num_layers': num_layers,
            'model_size_gb': model_size_gb,
            'hidden_size': getattr(config, 'hidden_size', None),
            'vocab_size': getattr(config, 'vocab_size', None),
            'architecture': config.model_type if hasattr(config, 'model_type') else 'unknown'
        }

    except Exception as e:
        print(f"[ERROR] Error fetching model info: {e}")
        return None


def get_hardware_info():
    """
    Query available hardware resources.

    Returns:
        dict: Available GPU, CPU RAM, and disk space
    """
    print("\n[INFO] Analyzing hardware...")

    hardware = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_devices': [],
        'system_ram_gb': 0,
        'disk_free_gb': 0
    }

    # GPU info
    if torch.cuda.is_available():
        hardware['gpu_count'] = torch.cuda.device_count()
        for i in range(hardware['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = props.total_memory / (1024**3)
            hardware['gpu_devices'].append({
                'id': i,
                'name': props.name,
                'total_memory_gb': gpu_memory_gb,
                'available_memory_gb': gpu_memory_gb * 0.9  # Reserve 10% for overhead
            })

    # System RAM
    ram = psutil.virtual_memory()
    hardware['system_ram_gb'] = ram.total / (1024**3)
    hardware['available_ram_gb'] = ram.available / (1024**3)

    # Disk space (current directory)
    disk = shutil.disk_usage('.')
    hardware['disk_free_gb'] = disk.free / (1024**3)

    return hardware


def calculate_allocation_strategies(model_info, hardware):
    """
    Calculate different memory allocation strategies.

    Returns:
        list: Different allocation strategies with predictions
    """
    print("\n[INFO] Calculating allocation strategies...")

    strategies = []

    model_size = model_info['model_size_gb']
    num_layers = model_info['num_layers']

    if num_layers is None:
        print("[WARNING] Cannot determine number of layers, providing basic recommendations only")
        return []

    # Calculate memory per layer (rough approximation)
    # Embeddings typically take ~10-15% of model size
    embedding_size = model_size * 0.12
    decoder_size = model_size - embedding_size
    memory_per_layer = decoder_size / num_layers

    # Strategy 1: CPU/Disk only (no GPU)
    strategies.append({
        'name': 'CPU/Disk Only (No GPU)',
        'description': 'All layers on CPU/Disk - slowest but works without GPU',
        'max_memory': {0: "0GiB", "cpu": f"{int(hardware['available_ram_gb'])}GiB"},
        'expected_distribution': {
            'gpu_layers': 0,
            'cpu_layers': min(num_layers, int(hardware['available_ram_gb'] / memory_per_layer)),
            'disk_layers': max(0, num_layers - int(hardware['available_ram_gb'] / memory_per_layer))
        },
        'feasible': hardware['available_ram_gb'] + hardware['disk_free_gb'] > model_size,
        'estimated_speed': 'Very Slow'
    })

    # Strategy 2-N: GPU + CPU + Disk (if GPU available)
    if hardware['gpu_available'] and len(hardware['gpu_devices']) > 0:
        gpu = hardware['gpu_devices'][0]
        gpu_memory = gpu['available_memory_gb']

        # Test different GPU allocations
        for gpu_alloc_pct in [0.5, 0.7, 0.9, 1.0]:
            gpu_alloc = gpu_memory * gpu_alloc_pct

            if gpu_alloc < 1:  # Skip if less than 1GB
                continue

            # Calculate layer distribution
            gpu_layers = int((gpu_alloc - embedding_size) / memory_per_layer) if gpu_alloc > embedding_size else 0
            gpu_layers = max(0, min(gpu_layers, num_layers))

            remaining_layers = num_layers - gpu_layers
            remaining_size = remaining_layers * memory_per_layer + (embedding_size if gpu_layers == 0 else 0)

            cpu_capacity = hardware['available_ram_gb']
            cpu_layers = int(cpu_capacity / memory_per_layer) if remaining_layers > 0 else 0
            cpu_layers = min(cpu_layers, remaining_layers)

            disk_layers = remaining_layers - cpu_layers

            # Determine speed estimate
            if gpu_layers >= num_layers * 0.8:
                speed = 'Fast'
            elif gpu_layers >= num_layers * 0.5:
                speed = 'Medium'
            elif gpu_layers > 0:
                speed = 'Slow'
            else:
                speed = 'Very Slow'

            strategies.append({
                'name': f'Balanced ({int(gpu_alloc)}GB GPU)',
                'description': f'{gpu_layers} layers on GPU, {cpu_layers} on CPU, {disk_layers} on disk',
                'max_memory': {0: f"{int(gpu_alloc)}GiB", "cpu": f"{int(cpu_capacity)}GiB"},
                'expected_distribution': {
                    'gpu_layers': gpu_layers,
                    'cpu_layers': cpu_layers,
                    'disk_layers': disk_layers
                },
                'feasible': (gpu_alloc + cpu_capacity + hardware['disk_free_gb']) > model_size,
                'estimated_speed': speed
            })

    return strategies


def print_report(model_info, hardware, strategies):
    """
    Print a formatted report of the analysis.
    """
    print("\n" + "="*70)
    print("MEMORY ALLOCATION PREDICTION REPORT")
    print("="*70)

    # Model info
    print(f"\n[MODEL] {model_info['model_name']}")
    print(f"   Architecture: {model_info['architecture']}")
    print(f"   Parameters: {model_info['num_parameters']:,}" if model_info['num_parameters'] else "   Parameters: Unknown")
    print(f"   Layers: {model_info['num_layers']}")
    print(f"   Model Size: {model_info['model_size_gb']:.2f} GB")

    # Hardware info
    print(f"\n[HARDWARE]")
    if hardware['gpu_available']:
        for gpu in hardware['gpu_devices']:
            print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
    else:
        print("   GPU: Not available")
    print(f"   System RAM: {hardware['system_ram_gb']:.1f} GB (Available: {hardware['available_ram_gb']:.1f} GB)")
    print(f"   Disk Space: {hardware['disk_free_gb']:.1f} GB free")

    # Strategies
    print(f"\n[RECOMMENDED STRATEGIES]")
    print("-" * 70)

    for i, strategy in enumerate(strategies, 1):
        status = "[FEASIBLE]" if strategy['feasible'] else "[NOT FEASIBLE]"
        print(f"\n{i}. {strategy['name']} - {status}")
        print(f"   {strategy['description']}")
        print(f"   Speed: {strategy['estimated_speed']}")
        print(f"   Config: max_memory={strategy['max_memory']}")

        dist = strategy['expected_distribution']
        print(f"   Distribution:")
        print(f"     - GPU: {dist['gpu_layers']} layers")
        print(f"     - CPU: {dist['cpu_layers']} layers")
        print(f"     - Disk: {dist['disk_layers']} layers")

    print("\n" + "="*70)
    print("\n[TIPS]")
    print("   - Use the 'max_memory' config in your script")
    print("   - Disk offloading is VERY slow - minimize disk layers if possible")
    print("   - Set offload_folder='./swap' for disk offloading")
    print("   - Consider using float16 or int8 quantization to reduce memory needs")
    print("="*70 + "\n")


def save_results(model_info, hardware, strategies, output_file='memory_analysis.json'):
    """
    Save analysis results to JSON file.
    """
    results = {
        'model_info': model_info,
        'hardware': hardware,
        'strategies': strategies
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Predict memory allocation for large language models')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name (e.g., facebook/opt-30b)')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--output', type=str, default='memory_analysis.json', help='Output JSON file')

    args = parser.parse_args()

    # Get model info
    model_info = get_model_info(args.model, args.token)
    if model_info is None:
        print("[ERROR] Failed to fetch model info. Exiting.")
        return

    # Get hardware info
    hardware = get_hardware_info()

    # Calculate strategies
    strategies = calculate_allocation_strategies(model_info, hardware)

    # Print report
    print_report(model_info, hardware, strategies)

    # Save results
    save_results(model_info, hardware, strategies, args.output)


if __name__ == "__main__":
    main()
