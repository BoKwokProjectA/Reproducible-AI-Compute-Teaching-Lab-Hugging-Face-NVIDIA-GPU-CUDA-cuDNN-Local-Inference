# Verified results

I recorded these outputs from two Google Colab sessions on 28–29 August 2026. Both sessions used an assigned Tesla T4.

## Environment

`python -m src.gpu_check`:

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

`nvidia-smi` showed driver 580.82.07, Tesla T4, 0 MiB / 15360 MiB, 0% utilisation and no running processes at the time of the check.

PyTorch reported 14913 MiB through `torch.cuda.mem_get_info()` rather than the 15360 MiB shown by `nvidia-smi`. About 450 MiB was already unavailable to the CUDA context.

Versions used for these runs: Transformers 4.57.6, Datasets 4.0.0, huggingface-hub 0.36.2, Python 3.13.15 and pytest 8.4.2.

## Dataset

Dataset revision:

`27aa014ce09b193e1a6f58112d4a66e0eddb69c5`

```
Classes (3): angular_leaf_spot, bean_rust, healthy
  train        1034 images
  validation   133 images
  test         128 images

First training image: (500, 500) px, label=angular_leaf_spot
```

Base model: `google/vit-base-patch16-224-in21k`

Model revision:

`b4569560a39a0f1af58e3ddaf17facf20ab919b0`

## Training

Command:

```bash
python -m src.train --epochs 3 --batch-size 16 --fp16
```

| Epoch | Train loss | Val accuracy |
|---|---|---|
| 1 | 0.4694 | 0.9699 |
| 2 | 0.1039 | 0.9624 |
| 3 | 0.0633 | 0.9323 |

**Test accuracy: 0.96875 in 41.7 seconds.** PyTorch reported 1348 MiB allocated and a 2365 MiB peak.

The first logged loss was 1.0164 at step 10. That is close to `ln(3) ≈ 1.099`, which was one of the checks I used after fixing the earlier ViT weight-loading issue.

Validation accuracy dropped after epoch 1 while training loss continued to fall. I treated that as mild overfitting rather than extending the run. The validation set contains 133 images, so one image changes accuracy by roughly 0.75 percentage points.

The corrected model load report showed only `classifier.bias` and `classifier.weight` as newly initialised. That was the expected result because the pretrained backbone should load while the three-class head starts fresh.

## Inference

GPU result:

```json
{
  "predicted_class": "angular_leaf_spot",
  "confidence": 0.884,
  "probabilities": {"angular_leaf_spot": 0.884, "bean_rust": 0.1035, "healthy": 0.0125},
  "device": "cuda:0",
  "using_gpu": true,
  "base_model_revision": "b4569560a39a0f1af58e3ddaf17facf20ab919b0",
  "dataset_revision": "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
}
```

Running the same image with `--cpu` produced the same class and the same confidence to four decimal places, with `"device": "cpu"` and `"using_gpu": false`.

The ground-truth label for that image was `angular_leaf_spot`.

Both paths used fp32 during inference. I only treat the matching values above as the result of this run; exact equality is not guaranteed on another machine.

## API

`GET /health` returned:

```json
{"status": "ok", "model_loaded": true, "model_error": null,
 "classes": ["angular_leaf_spot", "bean_rust", "healthy"],
 "device": "cuda:0", "cuda_available": true, "gpu_name": "Tesla T4",
 "torch_version": "2.11.0+cu128", "cudnn_version": 91900}
```

`POST /predict` with a real image returned **200** and the same prediction as the CLI.

I also sent text bytes with an `image/jpeg` content type. The API returned **400 `{'detail': 'Not a readable image'}`**, confirming that it attempted to decode the content rather than accepting the MIME type alone.

While the API was idle, `nvidia-smi` showed `Tesla T4, 0 %, 523 MiB, 15360 MiB`. The training peak was 2365 MiB.

## Tests

```
collected 9 items
tests/test_inference.py .........                                        [100%]
9 passed in 16.32s
```

Python 3.13.15, pytest 8.4.2.
