# Reproducible NVIDIA GPU and Hugging Face AI Compute Teaching Lab 

This project fine-tunes a Hugging Face Vision Transformer on the beans leaf-disease dataset, runs inference on GPU or CPU, and exposes the model through a small FastAPI service.

I used the project to work through the parts around the model that usually cause problems in practice: environment setup, GPU checks, repeatable dataset/model versions, training metadata, CPU fallback, testing, and a short student lab.

> **Status:** working. The results below came from Google Colab sessions on an NVIDIA Tesla T4. The recorded outputs are in [docs/session_results.md](docs/session_results.md).

## What is in the project

```
Hugging Face Hub                    src/dataset.py
  AI-Lab-Makerere/beans   ------>   load at pinned revision
  google/vit-base-...               processor-driven preprocessing
                                            |
                                            v
                                    src/train.py
                                    PyTorch loop, explicit device placement
                                    CUDA + cuDNN through PyTorch
                                            |
                                            v
                                    models/vit-beans/
                                    weights + processor + training_metadata.json
                                            |
                                            v
                                    src/inference.py
                                    BeanClassifier, GPU or CPU
                                            |
                                            v
                                    api/main.py
                                    GET /health, POST /predict
```

`src/gpu_check.py` is the common device check used by training, inference and the API health response.

## Stack

Python 3.13.15 · PyTorch 2.11 · Hugging Face Transformers 4.57 · Datasets 4.0 · Hugging Face Hub · FastAPI · Uvicorn · Pillow · pytest · Jupyter · Git

## Dataset and model

| | |
|---|---|
| Dataset | [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans) |
| Classes | angular_leaf_spot, bean_rust, healthy |
| Splits | train / validation / test, as published |
| Base model | [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) |

I resolve the current Hugging Face revisions with:

```bash
python -m src.dataset --show-revisions
```

The verified dataset and base-model commit SHAs are recorded in
`src/config.py`, preventing later runs from silently switching to newer
Hugging Face repository revisions.

The base checkpoint is the ImageNet-21k ViT model. Its original classification head does not match the three bean classes, so training replaces that head with a new 3-class layer.

## Quick start

### Google Colab

1. Open `notebook/student_gpu_ai_lab.ipynb`.
2. In Colab, select **Runtime > Change runtime type > GPU**.
3. Run the environment diagnostic cells.
4. Load the Hugging Face Beans dataset.
5. Load and fine-tune the ViT model.
6. Run GPU inference and inspect the recorded device information.
```


## GPU checks

I run `src/gpu_check.py` before training so I can see what PyTorch actually detects rather than assuming CUDA is available.

```bash
python -m src.gpu_check
python -m src.gpu_check --json
```

The check reports PyTorch, CUDA and cuDNN versions, NVIDIA driver information, CUDA availability, detected devices, compute capability and VRAM figures.

Recorded on the Tesla T4:

```
Environment
  PyTorch            2.11.0+cu128
  Built against CUDA 12.8
  cuDNN              91900 (enabled=True)
  NVIDIA driver      580.82.07

CUDA available: True
Selected device: cuda:0

Device 0: Tesla T4
  Compute capability  7.5
  Total VRAM          14913 MiB
  Free VRAM           14808 MiB
  Allocated by torch  0 MiB
  Reserved by torch   0 MiB
```

`nvidia-smi` showed 15360 MiB while `torch.cuda.mem_get_info()` showed 14913 MiB. The difference is memory already used by the CUDA context and driver.

The CUDA version shown by `nvidia-smi` is the maximum CUDA level supported by the host driver. PyTorch brings its own CUDA runtime and cuDNN, so the two version numbers do not have to be identical.

### What I check in `nvidia-smi`

I use the header for the driver/CUDA compatibility information, then check GPU name, memory use, utilisation and the process list. The process list is useful when an old notebook or crashed process is still holding VRAM.

PyTorch can also reserve freed memory in its caching allocator, so `nvidia-smi` may show more memory in use than `torch.cuda.memory_allocated()`.

## Training

Training uses a plain PyTorch loop. I kept it this way because I wanted device placement, mixed precision and memory use to be visible in the training code.

```bash
python -m src.train --epochs 1 --max-train-samples 64
python -m src.train --epochs 3 --batch-size 16 --fp16
```

A run saves the model, processor and `training_metadata.json`. The metadata includes the dataset/model revisions, hyperparameters, epoch metrics, test accuracy, duration, device details and library versions.

Recorded run on the Tesla T4:

| Epoch | Train loss | Val accuracy |
|---|---|---|
| 1 | 0.4694 | 0.9699 |
| 2 | 0.1039 | 0.9624 |
| 3 | 0.0633 | 0.9323 |

**Test accuracy: 0.96875 in 41.7 seconds.** The run used 1,034 training images. PyTorch reported 1348 MiB allocated and a 2365 MiB peak.

Validation accuracy was highest after epoch 1 while training loss kept falling, so the short run shows some overfitting. The validation set has only 133 images, which also makes the percentage move noticeably when only a few predictions change.

The opening loss was 1.0164, close to `ln(3) ≈ 1.099` for three classes. I used that as one of the checks after fixing an earlier model-loading problem. The failure and fix are recorded in [docs/GPU_TROUBLESHOOTING.md](docs/GPU_TROUBLESHOOTING.md).

## Inference

Training, inference and the API all use `select_device()` from `src/gpu_check.py`.

```bash
python -m src.inference leaf.jpg
python -m src.inference leaf.jpg --cpu
```

The result includes `device` and `using_gpu`, so it is clear which path ran.

GPU result:

```json
{"predicted_class": "angular_leaf_spot", "confidence": 0.884,
 "probabilities": {"angular_leaf_spot": 0.884, "bean_rust": 0.1035, "healthy": 0.0125},
 "device": "cuda:0", "using_gpu": true}
```

CPU result for the same image:

```json
{"predicted_class": "angular_leaf_spot", "confidence": 0.884,
 "probabilities": {"angular_leaf_spot": 0.884, "bean_rust": 0.1035, "healthy": 0.0125},
 "device": "cpu", "using_gpu": false}
```

The image label was `angular_leaf_spot`, so the prediction was correct. On this run the CPU and GPU confidence matched to four decimal places. I do not rely on exact numerical equality across every machine.

## API

The FastAPI service has two endpoints:

- `GET /health` reports model state, classes, device, CUDA availability, GPU name and library versions.
- `POST /predict` accepts an image, checks that Pillow can decode it, rejects empty uploads and files over 10 MB, and returns the class, confidence, probabilities, device and stored revisions.

The service can start without a checkpoint. In that case `/health` reports `degraded` and `/predict` returns 503.

Example health response from the Colab run:

```json
{"status": "ok", "model_loaded": true, "model_error": null,
 "classes": ["angular_leaf_spot", "bean_rust", "healthy"],
 "device": "cuda:0", "cuda_available": true, "gpu_name": "Tesla T4",
 "torch_version": "2.11.0+cu128", "cudnn_version": 91900}
```

I also sent text bytes while claiming `image/jpeg`; the API returned **400 `Not a readable image`**, which confirms that it checks the file contents instead of trusting the content-type header.

While the service was idle, `nvidia-smi` showed 523 MiB in use. The training peak was 2365 MiB because training also needs gradients, optimiser state and activations.


## Tests

```bash
python -m pytest tests/ -v
```

The tests cover device selection, a CPU-only diagnostics report, API health in loaded/degraded states, invalid and empty image uploads, prediction response structure, greyscale input, probability normalisation and GPU/CPU agreement when a checkpoint is available.

Checkpoint-dependent tests skip if `models/vit-beans/` is missing.

Recorded result:

```
collected 9 items
tests/test_inference.py .........                                        [100%]
9 passed in 16.32s
```

Python 3.13.15, pytest 8.4.2.

## Teaching notebooks

`notebook/student_gpu_ai_lab.ipynb` is a short guided lab covering `nvidia-smi`, CUDA/cuDNN checks, Hugging Face dataset/model loading, inference, VRAM monitoring and troubleshooting questions.

`notebook/colab_runner.ipynb` is the notebook I used to run the setup, training, inference, tests and API steps in order.

## Repository layout

```
├── src/
│   ├── config.py              IDs, revisions, paths
│   ├── gpu_check.py           diagnostics + device selection
│   ├── dataset.py             loading, revision resolution, collation
│   ├── train.py               fine-tuning + metadata capture
│   └── inference.py           BeanClassifier
├── api/main.py                FastAPI service
├── tests/test_inference.py
├── notebook/
│   ├── student_gpu_ai_lab.ipynb
│   └── colab_runner.ipynb
└── docs/
    ├── GPU_TROUBLESHOOTING.md
    └── REPRODUCIBILITY.md
```


## Licence

The dataset and base model have their own licences on the Hugging Face Hub. Check those licences before reuse.
