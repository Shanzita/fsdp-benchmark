"""Environment Info"""
import platform, subprocess, sys

def run_cmd(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except: return "N/A"

def main():
    print("=" * 60)
    print(" ENVIRONMENT REPORT")
    print("=" * 60)
    print("\n## Software")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Python:      {sys.version.split()[0]}")
    try:
        import torch
        print(f"  PyTorch:     {torch.__version__}")
        print(f"  CUDA:        {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"  GPU count:   {torch.cuda.device_count()}")
        else:
            print(f"  GPU:         Not available (login node)")
    except: print("  PyTorch:     NOT INSTALLED")
    try:
        import torchvision
        print(f"  TorchVision: {torchvision.__version__}")
    except: pass
    print(f"\n## Hardware")
    print(f"  CPU:         {run_cmd('lscpu | grep Model.name | cut -d: -f2 | xargs')}")
    print(f"  Cores:       {run_cmd('nproc')}")
    print(f"  RAM:         {run_cmd('free -g | awk /Mem:/{{print=$2}}')} GB")
    print(f"  NVIDIA Driver: {run_cmd('nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')}")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}:       {name} ({mem:.1f} GB)")
    except: pass
    print(f"\n## Distributed")
    try:
        import torch
        print(f"  NCCL:        {torch.distributed.is_nccl_available()}")
        print(f"  Gloo:        {torch.distributed.is_gloo_available()}")
    except: pass
    print("=" * 60)

if __name__ == "__main__":
    main()