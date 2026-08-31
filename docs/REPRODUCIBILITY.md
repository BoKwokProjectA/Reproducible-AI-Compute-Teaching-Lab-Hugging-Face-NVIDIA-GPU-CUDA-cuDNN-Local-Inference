# Reproducing this project

I ran the recorded project results in Google Colab, but the same code can be used on a local NVIDIA GPU machine. The main difference is how PyTorch and the GPU environment are installed.

## Recorded environment

| | Value |
|---|---|
| Developed on | Google Colab, NVIDIA Tesla T4 (compute 7.5, 15360 MiB) |
| PyTorch | 2.11.0+cu128 |
| CUDA (PyTorch build) | 12.8 |
| cuDNN | 91900 (9.19.0) |
| NVIDIA driver | 580.82.07 |
| Python | 3.13.15 (Colab); `environment.yml` targets 3.11 |
| transformers / datasets / hub | 4.57.6 / 4.0.0 / 0.36.2 |

I keep Transformers below 5.x because a Colab run with 5.15.1 did not load the ViT checkpoint correctly. That issue and the output I used to diagnose it are in [GPU_TROUBLESHOOTING.md](GPU_TROUBLESHOOTING.md).

## Prerequisites

- Git
- An NVIDIA GPU and working driver for the GPU path
- Around 5 GB free disk space for the environment, cached dataset and model files

A system-wide CUDA Toolkit is not required for the PyTorch setup used here. The PyTorch wheel includes its CUDA runtime and cuDNN; the host still needs the NVIDIA driver.

## 1. Clone

```bash
git clone <repository-url>
cd mmu-gpu-ai-lab
```

## 2. Create the environment

### Google Colab

I kept Colab's preinstalled PyTorch instead of replacing it, then installed the other project packages:

```bash
pip install -q "transformers<5" "datasets==4.0.0" "huggingface-hub<1.0" \
               "fastapi>=0.112" "uvicorn>=0.30" python-multipart httpx pytest requests
```

After changing package versions, restart the notebook kernel. I hit a stale-import problem when the kernel kept an older Transformers version in memory even though pip had changed the packages on disk.

`notebook/colab_runner.ipynb` follows the Colab steps in order.

## 3. Check the GPU

Before training, I run:

```bash
nvidia-smi
python -m src.gpu_check
```

If PyTorch does not report CUDA when a GPU is expected, I use the checks in `docs/GPU_TROUBLESHOOTING.md` before starting a run.

## 4. Record the Hugging Face revisions

```bash
python -m src.dataset --show-revisions
```

This prints the current commit SHAs for the dataset and base model. Store them in `src/config.py` as `DATASET_REVISION` and `BASE_MODEL_REVISION`.

Without a fixed revision, a later run would follow the current state of the Hugging Face repository. Recording the SHAs makes it possible to identify exactly which dataset and model checkpoint were used for a run.

I then check the dataset structure with:

```bash
python -m src.dataset --summary
```

## 5. Train

I use a short smoke test first so environment or model-loading problems show up before a full run:

```bash
python -m src.train --epochs 1 --batch-size 16 --max-train-samples 64
```

Recorded training command:

```bash
python -m src.train --epochs 3 --batch-size 16 --fp16
```

The output directory is `models/vit-beans/`. It contains the saved model, processor files and `training_metadata.json` with the revisions, hyperparameters, epoch metrics, device information and library versions.

`models/` is gitignored because the weights are large and can be rebuilt from the recorded configuration.

If training runs out of GPU memory, I would reduce the batch size to 8 and keep `--fp16` enabled.

## 6. Run inference

```bash
python -m src.inference path/to/leaf.jpg
python -m src.inference path/to/leaf.jpg --cpu
```

The second command forces the CPU path and is useful for checking the fallback behaviour.

## 7. Start the API

```bash
uvicorn api.main:app --reload
```

Then check:

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@leaf.jpg" http://127.0.0.1:8000/predict
```

FastAPI's interactive docs are at `http://127.0.0.1:8000/docs`.

The API can start even when the trained checkpoint is missing. In that state `/health` reports `degraded` and `/predict` returns 503.

## 8. Run the tests

```bash
python -m pytest tests/ -v
```

Tests that need `models/vit-beans/` skip when the checkpoint is not present, so the device and API validation tests can still run before training.

## Remaining reproducibility limits

- **Transitive dependencies:** `environment.yml` constrains the direct dependencies but is not a full lockfile.
- **Random seed:** training is not seeded, so the reported accuracy is one recorded run rather than a guaranteed repeatable number.
- **Colab GPU assignment:** Colab may assign a different accelerator in another session, so timing and memory figures may not be directly comparable.
