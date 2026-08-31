# GPU troubleshooting

I split this file into problems I actually hit during the Colab runs and GPU/PyTorch scenarios I documented for the lab. Items marked **[observed]** include problems from this project. Items marked **[scenario]** were not reproduced here.

## Problems I hit during the project

### 1. ViT weights did not load correctly after a Transformers major-version change **[observed]**

**What I saw**

Training completed, but validation accuracy was 0.3308 and test accuracy was 0.3359. With three balanced classes, that is effectively chance performance.

The model load report showed the encoder weights under two different naming layouts:

```
encoder.layer.{0...11}.attention.attention.query.weight | UNEXPECTED |
vit.layers.{0...11}.attention.q_proj.weight             | MISSING    |
```

The checkpoint used the Transformers v4 ViT layout while the Colab environment had installed **transformers 5.15.1**. The result was that the pretrained backbone was not loaded as expected.

`ignore_mismatched_sizes=True` is needed because this project replaces the original classifier with a three-class head. In this case it also meant the wider mismatch did not stop the run with an exception.

**What caused it**

My Colab install command used `transformers>=4.44`, so pip selected 5.15.1. The repository environment file did not have this problem.

**What I changed**

```bash
pip install "transformers<5"
```

That installed 4.57.6. `environment.yml` now keeps Transformers on the 4.57 minor release.

**How I checked the fix**

After the change, only `classifier.bias` and `classifier.weight` were reported as newly initialised. The opening loss was 1.0164, close to `ln(3) ≈ 1.099`, which was much more believable for a fresh three-class head.

This problem is why I also record software versions alongside the dataset and model revisions.

### 2. Training ran but INFO logs disappeared **[observed]**

**What I saw**

A run produced no normal training log output, including the first `Device: cuda:0` message, but the checkpoint and `training_metadata.json` were still written.

**What caused it**

Another imported library had already attached a handler to Python's root logger. `logging.basicConfig()` then did nothing, so the root logger stayed at WARNING and the project's `log.info(...)` messages were hidden.

**What I changed**

I changed the logging setup in `src/train.py` to use `logging.basicConfig(..., force=True)`.

**How I checked it**

Before changing the logger, `ls models/vit-beans` confirmed that the run had actually completed despite the empty console output. After the change, INFO messages appeared normally again.

### 3. Notebook kernel kept the old package version after pip downgrade **[observed]**

**What I saw**

After downgrading Transformers, an import inside the existing notebook kernel failed with:

```
ImportError: cannot import name 'is_offline_mode' from 'transformers.utils'
```

At the same time, `!python -m src.train` still worked.

**What caused it**

The notebook process still had modules from Transformers 5.15.1 in memory while the packages on disk had been changed to 4.57.6. A fresh subprocess read the new versions from disk, which is why the command-line training process behaved differently from the notebook kernel.

**Fix**

Restart the notebook kernel after changing package versions with pip.

---

## Checks for other GPU problems

These are the checks I included for the teaching/troubleshooting part of the project.

### Components to check

| Layer | What I check | Command / value |
|---|---|---|
| GPU | Whether the machine exposes an NVIDIA card | `nvidia-smi` |
| Driver | Whether the host driver is working | `nvidia-smi` header |
| CUDA runtime | Which CUDA build PyTorch is using | `torch.version.cuda` |
| cuDNN | Which cuDNN version PyTorch sees | `torch.backends.cudnn.version()` |

The CUDA number shown by `nvidia-smi` is the maximum CUDA level supported by the host driver. PyTorch can use a different CUDA runtime version bundled with its own wheel, as long as the driver supports it.

## PyTorch reports CUDA unavailable

If `torch.cuda.is_available()` is false when I expect a GPU, I check these in order.

### 1. Does the host see the GPU?

```bash
nvidia-smi
```

If that command fails, the problem is below PyTorch: no NVIDIA device, no working driver, or (on Colab) a CPU runtime was selected.

### 2. Is PyTorch CPU-only? **[scenario]**

```python
import torch
print(torch.version.cuda)
```

`None` means the installed PyTorch build has no CUDA runtime. On Colab I avoid reinstalling `torch` unless necessary because the supplied build is already matched to the runtime driver.

### 3. Is the PyTorch CUDA build newer than the driver supports? **[scenario]**

Compare the CUDA value in `nvidia-smi` with `torch.version.cuda`. If the PyTorch build requires a newer CUDA level than the driver supports, either use a compatible PyTorch build or update the driver where that is under your control.

### 4. Is the device hidden? **[scenario]**

An environment such as:

```bash
CUDA_VISIBLE_DEVICES=""
```

can hide an otherwise working GPU from the process. This is worth checking on shared machines.

`src/gpu_check.py` puts these details together in one report.

## CUDA out of memory

The first things I would check are:

- **Batch size.** VRAM use grows with batch size. For this project I would retry with `--batch-size 8`.
- **Training activations.** ViT-base weights alone are much smaller than the total memory required for backpropagation because training also keeps activations and gradients.
- **Precision.** `--fp16` reduces activation memory on a suitable GPU.
- **Tensors accidentally kept alive.** Accumulating a tensor such as `loss` can retain the computation graph. The training loop stores `loss.item()` instead.
- **Another process using VRAM.** I check the Processes section of `nvidia-smi`, especially after notebook crashes.

PyTorch's caching allocator may keep free blocks reserved. `torch.cuda.empty_cache()` can release unused cached blocks, but it does not fix an OOM if live tensors still require the memory.

## Version compatibility

For this project I treat the host driver and the PyTorch CUDA build as the important compatibility pair. The driver sets the highest CUDA level the machine supports, while the PyTorch wheel supplies its CUDA runtime and cuDNN. A PyTorch CUDA build therefore has to stay within the driver's supported range.
