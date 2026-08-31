"""GPU, CUDA and cuDNN diagnostics.

Run it as a script for a readable report:

    python -m src.gpu_check

or import collect_gpu_info() to put the same facts into the API health
response. Everything here reports what is actually present - if a value is
unavailable it comes back as None rather than a guess.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import torch


def select_device() -> torch.device:
    """The one place the GPU-or-CPU decision gets made.

    Training, inference and the API all call this, so they can't disagree
    about which device the model ended up on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _nvidia_smi(fields: list[str]) -> list[str] | None:
    """Query nvidia-smi, or return None if it isn't usable.

    nvidia-smi ships with the NVIDIA driver, not with PyTorch. If it's missing
    the problem is almost always the driver or the host, not the Python env -
    which is a useful thing to be able to distinguish when debugging.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_gpu_info() -> dict:
    """Gather everything worth knowing about the current compute environment."""
    info = {
        "torch_version": torch.__version__,
        # The CUDA version this PyTorch wheel was *built* against. Not the same
        # as the CUDA version the driver supports - see docs/GPU_TROUBLESHOOTING.md.
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "selected_device": str(select_device()),
        "driver_version": None,
        "devices": [],
    }

    driver = _nvidia_smi(["driver_version"])
    if driver:
        info["driver_version"] = driver[0]

    for index in range(info["device_count"]):
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        info["devices"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": round(total_bytes / 1024**2),
                "free_memory_mb": round(free_bytes / 1024**2),
                # allocated = tensors we asked for; reserved = what the caching
                # allocator is holding on to. Reserved is usually larger, and
                # that gap is what confuses people reading nvidia-smi.
                "allocated_mb": round(torch.cuda.memory_allocated(index) / 1024**2),
                "reserved_mb": round(torch.cuda.memory_reserved(index) / 1024**2),
            }
        )

    return info


def format_report(info: dict) -> str:
    lines = [
        "Environment",
        f"  PyTorch            {info['torch_version']}",
        f"  Built against CUDA {info['torch_cuda_build'] or 'n/a (CPU-only build)'}",
        f"  cuDNN              {info['cudnn_version'] or 'n/a'}"
        f" (enabled={info['cudnn_enabled']})",
        f"  NVIDIA driver      {info['driver_version'] or 'nvidia-smi not available'}",
        "",
        f"CUDA available: {info['cuda_available']}",
        f"Selected device: {info['selected_device']}",
    ]

    if not info["devices"]:
        lines += [
            "",
            "No CUDA devices visible. The project will run on CPU.",
            "If you expected a GPU here, start with docs/GPU_TROUBLESHOOTING.md.",
        ]
        return "\n".join(lines)

    for device in info["devices"]:
        lines += [
            "",
            f"Device {device['index']}: {device['name']}",
            f"  Compute capability  {device['compute_capability']}",
            f"  Total VRAM          {device['total_memory_mb']} MiB",
            f"  Free VRAM           {device['free_memory_mb']} MiB",
            f"  Allocated by torch  {device['allocated_mb']} MiB",
            f"  Reserved by torch   {device['reserved_mb']} MiB",
        ]

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report GPU/CUDA/cuDNN status.")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead"
    )
    args = parser.parse_args()

    info = collect_gpu_info()
    print(json.dumps(info, indent=2) if args.json else format_report(info))


if __name__ == "__main__":
    main()
