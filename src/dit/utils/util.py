import argparse


def parse_args_SiT():
    """Argument parser for JiT model scripts.

    Returns an argparse.Namespace with defaults matching the project's JiT Denoiser expectations.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", default=False, help="train model")
    argparser.add_argument("--sample", action="store_true", default=False, help="sample from model")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--dataset", type=str, default="gastronet", help="dataset name", choices=["gastronet"])
    argparser.add_argument("--batch_size", type=int, default=256, help="batch size")
    argparser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    argparser.add_argument("--model", type=str, default="SiT-B/2", help="SiT model variant (e.g. SiT-B/2, SiT-B/4)")
    argparser.add_argument(
        "--vae", type=str, default="SD2", help="VAE model to use (e.g. SD2)", choices=["SD2", "SD3", "Flux1", "Flux2"]
    )
    argparser.add_argument("--img_size", type=int, default=256, help="input image size")
    argparser.add_argument("--class_num", type=int, default=0, help="number of classes (for label embedding)")
    argparser.add_argument("--attn_dropout", type=float, default=0.0, help="attention dropout")
    argparser.add_argument("--proj_dropout", type=float, default=0.0, help="projection dropout")
    argparser.add_argument(
        "--label_drop_prob", type=float, default=0.1, help="probability to drop labels (classifier-free guidance)"
    )
    argparser.add_argument("--P_mean", type=float, default=-0.8, help="mean for timestep sampling (sigmoid space)")
    argparser.add_argument("--P_std", type=float, default=0.8, help="std for timestep sampling (sigmoid space)")
    argparser.add_argument("--t_eps", type=float, default=1e-5, help="epsilon for numerical stability in timesteps")
    argparser.add_argument("--ema_decay", type=float, default=0.9999, help="ema decay (fast)")
    argparser.add_argument(
        "--sampling_method", type=str, default="euler", choices=["euler", "heun"], help="ODE sampling method"
    )
    argparser.add_argument("--num_sampling_steps", type=int, default=50, help="number of sampling steps for generation")
    argparser.add_argument("--cfg", type=float, default=2.9, help="classifier-free guidance scale")
    argparser.add_argument("--interval_min", type=float, default=0.1, help="cfg interval min")
    argparser.add_argument("--interval_max", type=float, default=1.0, help="cfg interval max")
    argparser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    argparser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for Adam optimizer")
    argparser.add_argument("--snapshot", type=int, default=10, help="how many snapshots during training")
    argparser.add_argument("--no_wandb", action="store_true", default=False, help="Disable wandb logging")
    argparser.add_argument("--sample_and_save_freq", type=int, default=50, help="Sample and save frequency")
    argparser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps"
    )
    argparser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    argparser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    argparser.add_argument("--norm", action="store_true", default=False, help="Use ImageNet normalization")
    argparser.add_argument("--fid", action="store_true", default=False, help="Sample for FID evaluation")
    argparser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate for cosine scheduler")
    return argparser.parse_args()


def parse_args_iREPA():
    """Argument parser for JiT model scripts.

    Returns an argparse.Namespace with defaults matching the project's JiT Denoiser expectations.
    """
    return build_parser_iREPA().parse_args()


def build_parser_iREPA():
    """Builds the iREPA argument parser (without parsing), so other entrypoints can extend it."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", default=False, help="train model")
    argparser.add_argument("--sample", action="store_true", default=False, help="sample from model")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--dataset", type=str, default="gastronet", help="dataset name", choices=["gastronet"])
    argparser.add_argument("--batch_size", type=int, default=256, help="batch size")
    argparser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    argparser.add_argument("--model", type=str, default="SiT-B/2", help="SiT model variant (e.g. SiT-B/2, SiT-B/4)")
    argparser.add_argument(
        "--vae", type=str, default="SD2", help="VAE model to use (e.g. SD2)", choices=["SD2", "SD3", "Flux1", "Flux2"]
    )
    argparser.add_argument("--img_size", type=int, default=256, help="input image size")
    argparser.add_argument("--class_num", type=int, default=0, help="number of classes (for label embedding)")
    argparser.add_argument(
        "--label_drop_prob", type=float, default=0.1, help="probability to drop labels (classifier-free guidance)"
    )
    argparser.add_argument("--P_mean", type=float, default=-0.8, help="mean for timestep sampling (sigmoid space)")
    argparser.add_argument("--P_std", type=float, default=0.8, help="std for timestep sampling (sigmoid space)")
    argparser.add_argument("--t_eps", type=float, default=1e-5, help="epsilon for numerical stability in timesteps")
    argparser.add_argument("--ema_decay", type=float, default=0.9999, help="ema decay (fast)")
    argparser.add_argument(
        "--sampling_method", type=str, default="euler", choices=["euler", "heun"], help="ODE sampling method"
    )
    argparser.add_argument("--num_sampling_steps", type=int, default=50, help="number of sampling steps for generation")
    argparser.add_argument("--cfg", type=float, default=2.9, help="classifier-free guidance scale")
    argparser.add_argument("--interval_min", type=float, default=0.1, help="cfg interval min")
    argparser.add_argument("--interval_max", type=float, default=1.0, help="cfg interval max")
    argparser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    argparser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for Adam optimizer")
    argparser.add_argument("--snapshot", type=int, default=10, help="how many snapshots during training")
    argparser.add_argument("--no_wandb", action="store_true", default=False, help="Disable wandb logging")
    argparser.add_argument("--sample_and_save_freq", type=int, default=50, help="Sample and save frequency")
    argparser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps"
    )
    argparser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    argparser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    argparser.add_argument("--fid", action="store_true", default=False, help="Sample for FID evaluation")
    argparser.add_argument("--fused_attn", action=argparse.BooleanOptionalAction, default=True)
    argparser.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=False)
    argparser.add_argument("--projection_layer_type", type=str, default="conv", choices=["mlp", "conv"])
    argparser.add_argument(
        "--projection_loss_type",
        type=str,
        default="cosine",
        help="Should be a comma-separated list of projection loss types",
    )
    argparser.add_argument("--proj_kwargs_kernel_size", type=int, default=3, choices=[1, 3, 5, 7])
    argparser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    argparser.add_argument("--enc_type", type=str, default="dinov2-vit-b")
    argparser.add_argument("--proj_coeff", type=str, default="1.0")
    argparser.add_argument("--repa_loss", action=argparse.BooleanOptionalAction, default=True)
    argparser.add_argument("--spnorm_method", type=str, default="zscore", choices=["none", "zscore"])
    argparser.add_argument("--cls_token_weight", type=float, default=0.2)
    argparser.add_argument("--zscore_alpha", type=float, default=0.6)
    argparser.add_argument("--zscore_proj_skip_std", action=argparse.BooleanOptionalAction, default=False)
    argparser.add_argument("--encoder_depth", type=int, default=8)
    argparser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate for cosine scheduler")
    argparser.add_argument("--enc_ckpt_path", type=str, default=None, help="Path to encoder checkpoint")
    argparser.add_argument("--inpaint", action="store_true", default=False, help="inpaint using the model")
    argparser.add_argument("--outpaint", action="store_true", default=False, help="outpaint using the model")
    argparser.add_argument("--full", action="store_true", default=False, help="Train on full dataset")
    argparser.add_argument("--folder", type=str, default=None, help="Folder name for full dataset (if --full is set)")
    return argparser
