# KinDER + Nano World Model

Training and evaluating video world models on the [KinDER benchmark datasets](https://huggingface.co/datasets/kinder-bench/kinder-datasets) using the nano-world-model infrastructure.

---

## Setup

Download KinDER datasets from [🤗 Hugging Face](https://huggingface.co/datasets/kinder-bench/kinder-datasets).

Set the required environment variables before running any command:

```bash
# VAE weights (download once with: huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir ./pretrained_models/sd-vae-ft-mse/vae)
export VAE_MODEL_PATH=./pretrained_models/sd-vae-ft-mse

# Directory containing the KinDER HDF5 files (e.g. motion2d_p0.hdf5)
export DATASET_DIR=/path/to/kinder/data

# Where checkpoints and logs are saved
export RESULTS_DIR=/path/to/results
```

---

## Training

**Motion2D-p0** (NanoWM-B/2, 15k steps):

```bash
python src/main.py experiment=kinder_motion2d \
    dataset=kinder/motion2d model=nanowm_b2 \
    experiment.infra.mixed_precision=false
```

Checkpoints are saved under `${RESULTS_DIR}/<run_dir>/checkpoints/`.

---

## Evaluation

Replace the checkpoint path with your own trained checkpoint:

```bash
python src/main.py experiment=evaluate_only \
    dataset=kinder/motion2d model=nanowm_b2 \
    'experiment.resume_from_checkpoint="/path/to/checkpoints/across_timesteps/epoch=62-step=15000.ckpt"' \
    dataset.loader.validation_fixed_subset_size=64 \
    dataset.loader.validation_fixed_subset_seed=42 \
    experiment.infra.mixed_precision=false
```

> **Note:** The checkpoint path must be absolute (Hydra changes the working directory at runtime).
> Wrap it in single + double quotes to handle the `=` signs in the filename.

Evaluation outputs land under `${RESULTS_DIR}/<run_dir>/`:

```
<run_dir>/
├── eval_videos/        # GT vs predicted frame comparison MP4s
├── metrics.json        # PSNR / SSIM / LPIPS / FID
└── .hydra/             # Composed config snapshot
```
