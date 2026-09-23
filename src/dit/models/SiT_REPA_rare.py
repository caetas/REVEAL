"""iREPA fine-tuning on a labelled (rare) dataset, starting from an unconditional iREPA checkpoint,
plus an OOD setup: slightly noise an image, denoise it as "healthy", and compare SiT features of the
original and of the healthy reconstruction.
"""

import csv
import json
import os

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from config import models_dir, report_dir
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from .loss import SILoss
from .SiT_REPA import DenoiserREPA, SpatialNormalization, update_ema
from .vision_encoder import load_encoders

LABEL_TABLE_KEY = "y_embedder.embedding_table.weight"


def _strip_prefix(state_dict, prefix="module."):
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _auroc(scores, is_anomaly):
    """AUROC via the Mann-Whitney U statistic (ties get average ranks). Higher score = more anomalous."""
    scores = np.asarray(scores, dtype=np.float64)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    n_pos = int(is_anomaly.sum())
    n_neg = len(is_anomaly) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, first_idx, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = np.repeat(first_idx + (counts + 1) / 2.0, counts)
    return float((ranks[is_anomaly].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class DenoiserREPARare(DenoiserREPA):
    def __init__(self, args):
        assert args.class_num > 0 and args.label_drop_prob > 0, "fine-tuning needs classes and a null (CFG) label"
        super().__init__(args)
        self.healthy_label = args.healthy_label

    @property
    def null_label(self):
        # LabelEmbedder maps dropped labels (and the CFG unconditional branch) to index num_classes
        return self.num_classes

    ##################################################################################
    #                               Checkpoint loading                               #
    ##################################################################################

    def _load_matching(self, state_dict, set_manually=()):
        """Loads every key whose shape matches. Only projector keys may be skipped (they depend on the
        REPA encoder and are absent in eval mode), plus `set_manually` keys that the caller fills in itself;
        any other mismatch is an architecture error."""
        own = self.model.state_dict()
        loadable = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
        not_loaded = [k for k in own if k not in loadable and k not in set_manually]
        unexpected = [k for k in state_dict if k not in loadable]

        bad = [k for k in not_loaded + unexpected if "projectors" not in k]
        if bad:
            raise RuntimeError(f"Checkpoint does not match the model (missing/unexpected/mismatched keys): {bad}")

        self.model.load_state_dict(loadable, strict=False)
        return [k for k in not_loaded if "projectors" in k]

    def load_pretrained_unconditional(self, path):
        """Initialises the class-conditional model from an unconditional iREPA checkpoint.

        The unconditional model was trained with class_num=0 and label_drop_prob=1.0, so its label table
        has a single row (index 0), which is its null embedding. In the new model the null (CFG) label
        is index `num_classes`, so that row is copied there; the new class rows are initialised
        according to --class_init.
        """
        state_dict = _strip_prefix(torch.load(path, map_location="cpu"))
        pretrained_table = state_dict.pop(LABEL_TABLE_KEY)
        # the null embedding is always the last row (index = pretrained num_classes)
        null_embedding = pretrained_table[-1]
        if pretrained_table.shape[0] != 1:
            print(
                f"Warning: pretrained label table has {pretrained_table.shape[0]} rows (expected 1 for an "
                f"unconditional model); using its last row as the null embedding."
            )

        fresh_projectors = self._load_matching(state_dict, set_manually=(LABEL_TABLE_KEY,))

        with torch.no_grad():
            table = self.model.y_embedder.embedding_table.weight
            table[self.null_label] = null_embedding.to(table)
            if self.args.class_init == "null":
                table[: self.num_classes] = null_embedding.to(table)
            if self.args.class_init_noise > 0:
                table[: self.num_classes] += self.args.class_init_noise * torch.randn_like(table[: self.num_classes])

        print(f"Loaded unconditional checkpoint from {path}")
        print(f"  null embedding -> label index {self.null_label}; class rows 0..{self.num_classes - 1} "
              f"initialised with '{self.args.class_init}' (noise std {self.args.class_init_noise})")
        if fresh_projectors:
            print(f"  projectors not in checkpoint (or shape mismatch), freshly initialised: {len(fresh_projectors)} keys")

    def load_finetuned(self, path):
        """Loads a checkpoint produced by this fine-tuning (label table already has num_classes + 1 rows)."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        state_dict = _strip_prefix(torch.load(path, map_location="cpu"))
        n_rows = state_dict[LABEL_TABLE_KEY].shape[0]
        if n_rows != self.num_classes + 1:
            raise RuntimeError(
                f"Checkpoint label table has {n_rows} rows, expected {self.num_classes + 1} "
                f"(--class_num {self.num_classes} + null). Use --pretrained_checkpoint for unconditional checkpoints."
            )
        self._load_matching(state_dict)
        print(f"Loaded fine-tuned checkpoint from {path}")

    def load_for_finetuning(self):
        if self.args.checkpoint is not None:
            self.load_finetuned(self.args.checkpoint)
        else:
            self.load_pretrained_unconditional(self.args.pretrained_checkpoint)

    def load_checkpoint(self):
        if self.args.checkpoint is None:
            raise ValueError("--checkpoint (fine-tuned model) is required")
        self.load_finetuned(self.args.checkpoint)

    ##################################################################################
    #                                    Training                                    #
    ##################################################################################

    @torch.no_grad()
    def _encoder_targets(self, x, accelerator, spnorm):
        zs = []
        with accelerator.autocast():
            for encoder in self.encoders:
                encoder.eval()
                features = encoder.forward_features(encoder.preprocess(x * 127.5 + 127.5))
                zs.append(spnorm(feat=features["x_norm_patchtokens"], zscore_alpha=self.args.zscore_alpha))
        return zs

    def train_model(self, train_loader, verbose=True):
        """Fine-tunes the model on the labelled rare dataset (same objective as iREPA: velocity + REPA loss,
        with label dropout for CFG). Call `load_for_finetuning()` first."""
        run_name = f"{self.args.vae}_{self.args.model.replace('/', '_')}_{self.args.run_name}"
        accelerator = accelerate.Accelerator(
            log_with=None if self.no_wandb else "wandb",
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )
        if not self.no_wandb:
            accelerator.init_trackers("iREPA-rare", config=vars(self.args), init_kwargs={"wandb": {"name": run_name}})
        save_dir = os.path.join(models_dir, "iREPA_rare")
        os.makedirs(save_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.n_epochs * len(train_loader) // self.args.gradient_accumulation_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=self.lr / self.args.final_lr,
            final_div_factor=1,
            cycle_momentum=False,
        )

        train_loader, self.model, optimizer, scheduler, self.ema, self.vae = accelerator.prepare(
            train_loader, self.model, optimizer, scheduler, self.ema, self.vae
        )

        if self.args.repa_loss:
            self.encoders = load_encoders(
                self.args.enc_type,
                self.device,
                self.args.img_size,
                checkpoint_path=self.args.enc_ckpt_path,
                accelerator=accelerator,
            )

        loss_fn = SILoss(
            accelerator=accelerator,
            projection_loss_type=self.args.projection_loss_type,
            proj_coeff=self.args.proj_coeff,
        )
        spnorm = SpatialNormalization(self.args.spnorm_method)

        self.vae.eval()
        # the EMA starts from the (pretrained-initialised) model weights
        update_ema(self.ema, self.model, 0)

        snapshot_every = max(1, self.snapshot)
        n_samples = len(train_loader.dataset)

        for epoch in trange(self.n_epochs, desc="Epochs", disable=not verbose):
            self.model.train()
            train_loss = train_loss_vel = train_loss_proj = 0.0
            for x, y in tqdm(train_loader, desc="Batches", leave=False, disable=not verbose):
                with accelerator.accumulate(self.model):
                    y = y.long()
                    if x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    zs = self._encoder_targets(x, accelerator, spnorm)

                    with accelerator.autocast():
                        with torch.no_grad():
                            latents = self.encode(x)
                        loss_vel, loss_proj, _ = loss_fn(self.model, latents, dict(y=y), zs=zs)
                        loss_vel = loss_vel.mean()
                        loss_proj = loss_proj.mean()
                        loss = loss_vel + loss_proj

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() * x.shape[0]
                    train_loss_vel += loss_vel.item() * x.shape[0]
                    train_loss_proj += loss_proj.item() * x.shape[0]
                if accelerator.sync_gradients:
                    update_ema(self.ema, self.model, self.ema_decay)

            accelerator.wait_for_everyone()

            if not self.no_wandb:
                accelerator.log(
                    {
                        "train_loss": train_loss / n_samples,
                        "train_loss_vel": train_loss_vel / n_samples,
                        "train_loss_proj": train_loss_proj / n_samples,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=epoch,
                )

            if (epoch + 1) % snapshot_every == 0 or (epoch + 1) == self.n_epochs:
                accelerator.save(
                    accelerator.unwrap_model(self.ema).state_dict(),
                    os.path.join(save_dir, f"{run_name}_epoch{epoch + 1}.pt"),
                )

            if epoch == 0 or (epoch + 1) % self.sample_and_save_freq == 0:
                self.model.eval()
                sample_labels = torch.arange(0, 16, device=accelerator.device) % self.num_classes
                samples = self.generate(sample_labels, accelerator=accelerator).float()
                samples = torch.clamp(samples * 0.5 + 0.5, 0.0, 1.0)
                fig = plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.imshow(make_grid(samples, nrow=4).cpu().permute(1, 2, 0).numpy())
                if not self.no_wandb:
                    accelerator.log({"samples": fig}, step=epoch)
                plt.close(fig)
                self.model.train()

        accelerator.end_training()

    ##################################################################################
    #                                 OOD evaluation                                 #
    ##################################################################################

    @torch.no_grad()
    def encode_deterministic(self, x, accelerator):
        """VAE encoding with the posterior mode (not a sample), so OOD scores are reproducible."""
        vae = accelerator.unwrap_model(self.vae)
        dist = vae.encode(x).latent_dist
        if self.args.vae == "Flux2":
            vae.bn.eval()
            return vae.bn(self._patchify_latents(dist.mode()))
        return (dist.mode() - self.bias) * self.scale

    @torch.no_grad()
    def reconstruct_healthy(self, z1, accelerator, generator=None):
        """Noises clean latents to t0 = 1 - noise_level and integrates the ODE back to t=1 conditioned on the
        healthy label (with CFG, using --cfg / --interval_min / --interval_max)."""
        bsz = z1.size(0)
        t0 = 1.0 - self.args.ood_noise_level
        eps = torch.randn(z1.shape, generator=generator, device=z1.device, dtype=z1.dtype)
        z = t0 * z1 + (1.0 - t0) * eps

        labels = torch.full((bsz,), self.healthy_label, device=z1.device, dtype=torch.long)
        timesteps = (
            torch.linspace(t0, 1.0, self.args.ood_steps + 1, device=z1.device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, bsz, -1, -1, -1)
        )
        stepper = self._euler_step if self.method == "euler" else self._heun_step

        with accelerator.autocast():
            for i in range(self.args.ood_steps - 1):
                z = stepper(z, timesteps[i], timesteps[i + 1], labels)
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z.float()

    @torch.no_grad()
    def extract_features(self, z, accelerator, depths, eps=None):
        """SiT token features [B, T, D] at the given block depths, at timestep --ood_feat_t.
        `eps` must be shared between the inputs being compared so the noise cancels out."""
        model = accelerator.unwrap_model(self.model)
        bsz = z.size(0)
        t = self.args.ood_feat_t
        if t < 1.0:
            z = t * z + (1.0 - t) * eps
        label = self.null_label if self.args.ood_feat_label == "null" else self.healthy_label
        labels = torch.full((bsz,), label, device=z.device, dtype=torch.long)
        tt = torch.full((bsz,), t, device=z.device, dtype=z.dtype)
        with accelerator.autocast():
            feats = model.forward_features(z, tt, labels, encoder_depths=depths, proj=False)
        return [f.float() for f in feats]

    def _feature_depths(self):
        if self.args.ood_feat_depths:
            return sorted(int(d) for d in self.args.ood_feat_depths.split(","))
        return [self.args.encoder_depth]

    @torch.no_grad()
    def ood_evaluate(self, dataloader):
        """Healthy-reconstruction OOD evaluation.

        For each image: encode -> noise to t0 = 1 - ood_noise_level -> denoise with the healthy label ->
        compare SiT features of the original and of the reconstruction (per-token cosine distance).
        Also reports latent and pixel reconstruction errors. Labels != --healthy_label count as anomalies.
        Writes per-sample scores (CSV), AUROC per score (JSON) and a preview figure.
        """
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            raise RuntimeError("OOD evaluation runs on a single process; launch without --multi_gpu")
        self.model, self.vae = accelerator.prepare(self.model, self.vae)
        self.model.eval()
        self.vae.eval()
        device = accelerator.device

        depths = self._feature_depths()
        ckpt_name = os.path.splitext(os.path.basename(self.args.checkpoint))[0]
        out_dir = os.path.join(
            report_dir,
            "ood",
            f"{ckpt_name}_noise{self.args.ood_noise_level}_steps{self.args.ood_steps}_{self.method}"
            f"_cfg{self.cfg_scale}_featt{self.args.ood_feat_t}_{self.args.ood_feat_label}",
        )
        os.makedirs(out_dir, exist_ok=True)

        generator = torch.Generator(device=device).manual_seed(self.args.ood_seed)
        # (path, label, center) per sample, if the dataset exposes it (RareDataset does); needs shuffle=False
        samples = getattr(dataloader.dataset, "samples", None)
        rows = []
        preview = None

        for x, y in tqdm(dataloader, desc="OOD"):
            x = x.to(device)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            y = y.long()

            with accelerator.autocast():
                z1 = self.encode_deterministic(x, accelerator).float()
            z_rec = self.reconstruct_healthy(z1, accelerator, generator=generator)

            feat_eps = torch.randn(z1.shape, generator=generator, device=device, dtype=z1.dtype)
            feats_orig = self.extract_features(z1, accelerator, depths, eps=feat_eps)
            feats_rec = self.extract_features(z_rec, accelerator, depths, eps=feat_eps)

            with accelerator.autocast():
                # compare against the VAE reconstruction of the original to factor out VAE error
                x_vae = self.decode(z1).float()
                x_rec = self.decode(z_rec).float()

            scores = {
                "latent_mse": ((z1 - z_rec) ** 2).flatten(1).mean(1),
                "pixel_mse": ((x_vae - x_rec) ** 2).flatten(1).mean(1),
            }
            maps = {}
            for d, fo, fr in zip(depths, feats_orig, feats_rec):
                dist = 1.0 - F.cosine_similarity(fo, fr, dim=-1)  # [B, T]
                maps[d] = dist
                scores[f"feat_cos_d{d}_mean"] = dist.mean(1)
                scores[f"feat_cos_d{d}_max"] = dist.amax(1)
            if len(depths) > 1:
                scores["feat_cos_all_mean"] = torch.stack([scores[f"feat_cos_d{d}_mean"] for d in depths]).mean(0)

            for i in range(x.size(0)):
                row = {
                    "index": len(rows),
                    "label": int(y[i]),
                    "is_anomaly": int(y[i] != self.healthy_label),
                }
                if samples is not None:
                    row["path"], _, row["center"] = samples[row["index"]]
                row.update({k: float(v[i]) for k, v in scores.items()})
                rows.append(row)

            if preview is None:
                preview = (x_vae.cpu(), x_rec.cpu(), maps[depths[0]].cpu(), y.cpu())

        with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        score_names = [k for k in rows[0] if k not in ("index", "label", "is_anomaly", "path", "center")]
        is_anomaly = [r["is_anomaly"] for r in rows]
        summary = {
            "n_samples": len(rows),
            "n_anomalous": int(sum(is_anomaly)),
            "auroc": {k: _auroc([r[k] for r in rows], is_anomaly) for k in score_names},
            "args": vars(self.args),
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        self._save_ood_preview(*preview, depth=depths[0], path=os.path.join(out_dir, "preview.png"))

        print(f"OOD results saved to {out_dir}")
        for k, v in summary["auroc"].items():
            print(f"  AUROC {k}: {v:.4f}")
        return summary

    def _save_ood_preview(self, x_vae, x_rec, feat_map, labels, depth, path):
        """Rows: original (VAE recon), healthy reconstruction, |pixel diff|, feature-distance map."""
        n = min(self.args.ood_num_vis, x_vae.size(0))
        x_vae = torch.clamp(x_vae[:n] * 0.5 + 0.5, 0, 1)
        x_rec = torch.clamp(x_rec[:n] * 0.5 + 0.5, 0, 1)
        diff = (x_vae - x_rec).abs().mean(1)

        side = int(feat_map.shape[1] ** 0.5)
        feat_map = feat_map[:n].reshape(n, 1, side, side)
        feat_map = F.interpolate(feat_map, size=x_vae.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

        fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
        rows = [
            ("input", x_vae.permute(0, 2, 3, 1), None),
            ("healthy recon", x_rec.permute(0, 2, 3, 1), None),
            ("|pixel diff|", diff, "magma"),
            (f"feat dist d{depth}", feat_map, "magma"),
        ]
        for r, (title, imgs, cmap) in enumerate(rows):
            for c in range(n):
                ax = axes[r, c]
                ax.imshow(imgs[c].numpy(), cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(title)
                if r == 0:
                    ax.set_title(f"label {int(labels[c])}", fontsize=9)
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
