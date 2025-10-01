"""
Quick compatibility test for the memory_predictor script.
Tests all imports and basic functionality.
"""

print("Testing imports...")

try:
    import sys
    print(f"Python version: {sys.version}")
except Exception as e:
    print(f"[ERROR] Python: {e}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
except Exception as e:
    print(f"[ERROR] PyTorch/CUDA: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"[ERROR] Transformers: {e}")

try:
    import accelerate
    print(f"Accelerate version: {accelerate.__version__}")
except Exception as e:
    print(f"[ERROR] Accelerate: {e}")

try:
    from huggingface_hub import HfApi
    print(f"HuggingFace Hub: OK")
except Exception as e:
    print(f"[ERROR] HuggingFace Hub: {e}")

try:
    import psutil
    print(f"psutil version: {psutil.__version__}")
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.total / (1024**3):.1f} GB (Available: {ram.available / (1024**3):.1f} GB)")
except Exception as e:
    print(f"[ERROR] psutil: {e}")

try:
    import shutil
    disk = shutil.disk_usage('.')
    print(f"Disk free: {disk.free / (1024**3):.1f} GB")
except Exception as e:
    print(f"[ERROR] shutil: {e}")

print("\n" + "="*60)
print("Testing memory_predictor functions...")

try:
    from memory_predictor import get_hardware_info
    hardware = get_hardware_info()
    print("get_hardware_info(): OK")
    print(f"  Detected {hardware['gpu_count']} GPU(s)")
    print(f"  System RAM: {hardware['system_ram_gb']:.1f} GB")
except Exception as e:
    print(f"[ERROR] get_hardware_info(): {e}")

print("\n" + "="*60)
print("All compatibility checks complete!")
