"""Environment Info — For report Sections 3 & 4"""
import platform, subprocess, sys

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "N/A"

def main():
    print("=" * 60)
    print(" ENVIRONMENT REPORT")
    print("=" * 60)
    print(f"\n## Software")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Python:      {sys.version.split()[0]}")
    try:
        import torch
        print(f"  PyTorch:     {torch.__version__}")
        print(f"  CUDA:        {torch.version.cuda or 'N/A'}")
        if torch.cuda.is_available():
            print(f"  GPU count:   {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                print(f"  GPU {i}:       {name} ({mem:.1f} GB)")
        else:
            print("  GPU:         Not available (login node)")
    except ImportError:
        print("  PyTorch:     NOT INSTALLED")
    try:
        import torchvision
        print(f"  TorchVision: {torchvision.__version__}")
    except ImportError:
        pass

    print(f"\n## Hardware")
    print(f"  CPU:         {run_cmd('lscpu | grep Model.name | cut -d: -f2 | xargs')}")
    print(f"  Cores:       {run_cmd('nproc')}")
    print(f"  RAM:         {run_cmd('free -g | awk /Mem/{print($2)}')} GB")
    
    nvidia_smi = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1")
    print(f"  NVIDIA Driver: {nvidia_smi or 'N/A (login node)'}")

    print(f"\n## Distributed")
    try:
        import torch
        print(f"  NCCL:        {torch.distributed.is_nccl_available()}")
        print(f"  Gloo:        {torch.distributed.is_gloo_available()}")
    except:
        pass
    print("=" * 60)

if __name__ == "__main__":
    main()