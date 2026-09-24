![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## [DCA-MI 2026 (ECCV Workshop)] - REVEAL

<p align="center">
  <img src="imgs/reveal.png" width="100%" alt='Generated samples.'>
</p>

The official implementation of [**Representation-driven Endoscopic Visual
Embedding Alignment for Latent Generation**](https://arxiv.org/abs/2608.07176).

**[Francisco Caetano](https://caetas.github.io)<sup>1</sup>, [Tim J.M. Jaspers](https://scholar.google.com/citations?user=nwfiV2wAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Haiko Middeljans](https://scholar.google.com/citations?user=c4t8jsQAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Martijn R. Jong](https://scholar.google.com/citations?user=QRNrL-oAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Rixta A.H. van Eijck van Heslinga](https://pure.amsterdamumc.nl/en/persons/rixta-van-eijck-van-heslinga/)<sup>2</sup>, [Floor Slooter](https://amsterdamumc.org/en/research/researchers/floor-slooter.htm)<sup>2</sup>, [Albert Jeroen de Groof](https://scholar.google.com/citations?user=nT3VfE4AAAAJ&hl=en&oi=ao)<sup>2</sup>, [Jacques J. Bergman](https://scholar.google.com/citations?user=4SFBE0IAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Peter H.N. de With](https://www.tue.nl/en/research/researchers/peter-de-with)<sup>1</sup>, [Fons van der Sommen](https://scholar.google.com/citations?user=qFiLkCAAAAAJ&hl=en&oi=ao)<sup>1</sup>**

<sup>1</sup> Eindhoven University of Technology, 
<sup>2</sup> Amsterdam University Medical Centers

## What is REVEAL?

Developing foundation generative models for endoscopy is limited by the gap between natural and clinical images and the computational cost of training large Diffusion Transformers. Although representation alignment has improved efficiency in general computer vision, its role within the highly specialized endoscopic image space remains unclear. We introduce REVEAL (Representation-driven Endoscopic Visual Embedding Alignment), a foundation generative model trained on GastroNet-5M, a multicenter corpus of 5 million endoscopic frames. Instead of depending on out-of-domain priors, REVEAL employs encoders pretrained directly on the endoscopic distribution to align diffusion latents with domain-specific visual features, preserving fine textures and intricate anatomical structures. Beyond image generation, REVEAL also serves as a strong feature extractor: on the POLAR and Barrett’s Esophagus benchmarks, its internal representations show greater semantic richness than current specialized endoscopic models. REVEAL produces high-fidelity images and maintains robust structural coherence in latent-space edits such as inpainting and outpainting. This high-capacity backbone lowers the computational threshold for building specialized clinical tools, offering an open, versatile foundation for future intelligent gastroenterology systems.

## Repository Structure

- `src/dit/`: SiT and iREPA training/sampling entrypoints
- `src/jit/`: JiT training/sampling entrypoints
- `data/raw/GastroNet-5M/`: dataset location expected by dataloaders
- `data/raw/RARE25-train-data/`: labelled RARE25 training data (`<center>/{ndbe,neo}/*.png`) for fine-tuning
- `data/raw/RARE25-val-data/`: labelled RARE25 validation data (`{ndbe,neo}/*.png`) for OOD evaluation
- `models/iREPA/`: iREPA checkpoints (includes one example checkpoint)
- `models/iREPA_rare/`: checkpoints of iREPA fine-tuned on RARE25
- `models/pretrained_models/`: pretrained encoder/model checkpoints

## Setup

### Prerequisites

- `python>=3.12` (see `pyproject.toml`)
- `uv`
- `git`
- `NVIDIA Drivers`(mandatory) and `CUDA >= 12.8` (mandatory if Docker/Apptainer is not used)
- `Weights & Biases` account

### Installation (uv)

```bash
git clone git@github.com:caetas/REVEAL.git
cd reveal
uv sync --python 3.12
```

#### Environment Variables

This project reads paths from `.env` (already present in the repository template).

- dataset root: `DIR_DATA_RAW`
- model root: `DIR_MODELS`

If you use Weights & Biases, create a `.secrets` file with:

```bash
WANDB_API_KEY=<your-wandb-api-key>
```
### Installation (Docker/Apptainer)

Convert the Docker Image to a `.sif` file:

    apptainer pull reveal.sif docker://ocaetas/reveal

Then run the script [`job_apptainer.sh`](scripts/job_apptainer.sh) that will execute [`main.sh`](scripts/main.sh):
    
    cd scripts
    bash job_apptainer.sh

To access the shell, please run:

    apptainer shell --nv --env-file .env --bind $(pwd)/:/app/ reveal.sif

**Add the flag `--nvccli` if you are using WSL.**

**Note: Edit the [`main.sh`](scripts/main.sh) script if you want to train a different model.**

## Training

### Dataset Download

The full dataset can be downloaded [`here`](https://cortex.thetavision.nl/dataset-provider/listing/1/).

### Pretrained Encoders

You need to [request access](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) to access the DINOv3 pretrained encoders.

The GastroNet-5M pretrained encoders can be downloaded [`here`](https://cortex.thetavision.nl/dataset-provider/listing/2/).

### iREPA training (`src/dit/iREPA.py`)

```bash
cd src/dit
uv run accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes=4 iREPA.py \
  --train \
  --dataset gastronet \
  --img_size 256 \
  --model SiT-L/2 \
  --class_num 0 \
  --batch_size 128 \
  --n_epochs 63 \
  --sample_and_save_freq 7 \
  --ema_decay 0.9996 \
  --num_workers 64 \
  --lr 2e-4 \
  --final_lr 5e-5 \
  --vae SD2 \
  --enc_type dinov3-vit-b16 \
  --snapshot 1 \
  --enc_ckpt_path ./../../models/pretrained_models/gastro_231k.pth \
  --gradient_accumulation_steps 2 \
  --full
```

## Inference

### 1) Download pretrained weights

The folder containing the pretrained weights of the models used in the paper can be downloaded [`here`](https://huggingface.co/ocaetas/REVEAL).

### 2) Run sampling with iREPA

```bash
cd src/dit
uv run accelerate launch --mixed_precision=bf16 iREPA.py \
  --sample \
  --img_size 256 \
  --model SiT-L/2 \
  --class_num 0 \
  --vae SD2 \
  --enc_type dinov3-vit-b16 \
  --checkpoint ../../models/iREPA/SD2_SiT-L_2_gastronet.pt \
  --num_samples 16
```

## Fine-tuning on RARE25 and OOD detection (`src/dit/iREPA_rare.py`)

The unconditional iREPA checkpoint is fine-tuned as a class-conditional model on RARE25
(`ndbe` = 0, `neo` = 1). The pretrained unconditional embedding is used as the null label for
classifier-free guidance (index `--class_num`), and the new class embeddings are initialised from it
(`--class_init null`, or `random`).

### 1) Fine-tune from the unconditional checkpoint

```bash
cd src/dit
uv run accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes=4 iREPA_rare.py \
  --train \
  --img_size 256 \
  --model SiT-L/2 \
  --class_num 2 \
  --vae SD2 \
  --enc_type dinov3-vit-b16 \
  --enc_ckpt_path ./../../models/pretrained_models/gastro_231k.pth \
  --pretrained_checkpoint ../../models/iREPA/SD2_SiT-L_2_gastronet.pt \
  --batch_size 32 \
  --n_epochs 100 \
  --snapshot 10 \
  --sample_and_save_freq 10 \
  --ema_decay 0.999 \
  --lr 5e-5 \
  --final_lr 1e-6 \
  --num_workers 16
```

Checkpoints (EMA) are saved to `models/iREPA_rare/SD2_SiT-L_2_<run_name>_epoch<N>.pt`.
Useful options: `--rare_balanced` (class-balanced sampling, NEO is ~5% of the data),
`--rare_centers center_1` (train on a subset of centers), `--run_name`,
`--checkpoint <fine-tuned.pt>` instead of `--pretrained_checkpoint` to resume.

### 2) Sample per class

```bash
cd src/dit
uv run accelerate launch --mixed_precision=bf16 iREPA_rare.py \
  --sample \
  --img_size 256 \
  --model SiT-L/2 \
  --class_num 2 \
  --vae SD2 \
  --checkpoint ../../models/iREPA_rare/SD2_SiT-L_2_rare_epoch100.pt \
  --cfg 2.9 \
  --num_samples 16
```

### 3) OOD evaluation (healthy reconstruction)

Each image is slightly noised (`--ood_noise_level`, the ODE starts at `t = 1 - noise_level`),
denoised as healthy (`--healthy_label 0`, NDBE), and the SiT features of the original and the
reconstruction are compared (per-token cosine distance at `--ood_feat_depths`). Latent and pixel
reconstruction errors are reported too. Evaluation uses `data/raw/RARE25-val-data` (override with
`--rare_val_root`). Runs on a single GPU.

```bash
cd src/dit
uv run accelerate launch --mixed_precision=bf16 --num_processes=1 iREPA_rare.py \
  --ood \
  --img_size 256 \
  --model SiT-L/2 \
  --class_num 2 \
  --vae SD2 \
  --checkpoint ../../models/iREPA_rare/SD2_SiT-L_2_rare_epoch100.pt \
  --healthy_label 0 \
  --ood_noise_level 0.2 \
  --ood_steps 10 \
  --ood_feat_depths 4,8,12 \
  --ood_feat_t 1.0 \
  --cfg 2.9 \
  --batch_size 32
```

Results are written to `reports/ood/<checkpoint>_<settings>/`: `scores.csv` (per-image scores, with
path and hospital, taken from the filename prefix), `summary.json` (per score: AUROC and PPV at 90% recall,
NEO = positive; change the recall with `--ood_target_recall`) and `preview.png`.

## Citation

If you use this codebase, please cite:

```bibtex
@misc{caetano2026representation,
      title={Representation-driven Endoscopic Visual Embedding Alignment for Latent Generation}, 
      author={Francisco Caetano and Tim J. M. Jaspers and Haiko Middeljans and Martijn R. Jong and Rixta A. H. van Eijck van Heslinga and Floor Slooter and Albert J. de Groof and Jacques J. Bergman and Peter H. N. De With and Fons van der Sommen},
      year={2026},
      eprint={2608.07176},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.07176}, 
}
```

## License

This project is licensed under the terms of a custom license. See [LICENSE](LICENSE).
