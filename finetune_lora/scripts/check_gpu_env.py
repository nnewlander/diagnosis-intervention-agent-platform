"""检查 PyTorch CUDA 是否可用，便于训练前自检。"""
from __future__ import annotations

import sys

import torch


def main() -> None:
    ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {ok}")
    if ok:
        n = torch.cuda.device_count()
        print(f"device_count: {n}")
        for i in range(n):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  [0] total_memory_GiB: {props.total_memory / (1024**3):.2f}")
    else:
        print("未检测到可用 CUDA 设备。请检查 NVIDIA 驱动与安装了 CUDA 版 PyTorch。")
        sys.exit(1)


if __name__ == "__main__":
    main()
