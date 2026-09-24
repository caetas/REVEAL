from utils.util import build_parser_iREPA


def parse_args_iREPA_rare():
    """Argument parser for iREPA fine-tuning on the rare (labelled) dataset and OOD evaluation.

    Extends the iREPA parser, so every iREPA flag (model, vae, encoder, REPA loss, sampling, ...) is still available.
    """
    argparser = build_parser_iREPA()

    group = argparser.add_argument_group("rare fine-tuning")
    group.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to the unconditional iREPA (EMA) checkpoint used to initialise fine-tuning",
    )
    group.add_argument(
        "--class_init",
        type=str,
        default="null",
        choices=["null", "random"],
        help="Init of the new class embeddings: copy the pretrained null embedding, or random N(0, 0.02)",
    )
    group.add_argument(
        "--class_init_noise", type=float, default=0.0, help="Std of Gaussian noise added to the new class embeddings"
    )
    group.add_argument("--run_name", type=str, default="rare", help="Name used for checkpoints and wandb runs")

    group = argparser.add_argument_group("rare dataset")
    group.add_argument(
        "--rare_root", type=str, default=None, help="Training data root (default: data/raw/RARE25-train-data)"
    )
    group.add_argument(
        "--rare_val_root", type=str, default=None, help="Validation data root (default: data/raw/RARE25-val-data)"
    )
    group.add_argument(
        "--rare_centers", type=str, default=None, help="Comma-separated centers to use, e.g. center_1 (default: all)"
    )
    group.add_argument(
        "--rare_balanced", action="store_true", default=False, help="Class-balanced sampling during fine-tuning"
    )

    group = argparser.add_argument_group("OOD evaluation")
    group.add_argument("--ood", action="store_true", default=False, help="run OOD evaluation")
    group.add_argument("--healthy_label", type=int, default=0, help="Class index of the healthy class")
    group.add_argument(
        "--ood_noise_level",
        type=float,
        default=0.2,
        help="How much noise is added before reconstruction, in (0, 1]; the ODE starts at t0 = 1 - noise_level",
    )
    group.add_argument(
        "--ood_steps", type=int, default=10, help="Number of ODE steps from t0 to 1 for the healthy reconstruction"
    )
    group.add_argument(
        "--ood_feat_depths",
        type=str,
        default=None,
        help="Comma-separated SiT block depths to extract features from (default: --encoder_depth)",
    )
    group.add_argument(
        "--ood_feat_label",
        type=str,
        default="null",
        choices=["null", "healthy"],
        help="Label used when extracting features: the unconditional (null) label or the healthy label",
    )
    group.add_argument("--ood_seed", type=int, default=0, help="Seed for the OOD noise")
    group.add_argument(
        "--ood_target_recall", type=float, default=0.9, help="Recall at which PPV is reported (PPV@recall)"
    )
    group.add_argument("--ood_num_vis", type=int, default=8, help="Number of samples shown in the preview figure")

    args = argparser.parse_args()

    if args.class_num < 1:
        argparser.error("--class_num must be >= 1 (number of classes in the rare dataset)")
    if args.label_drop_prob <= 0:
        argparser.error("--label_drop_prob must be > 0, otherwise there is no null (CFG) embedding")
    if not 0 <= args.healthy_label < args.class_num:
        argparser.error(f"--healthy_label must be in [0, {args.class_num})")
    if not 0 < args.ood_noise_level <= 1:
        argparser.error("--ood_noise_level must be in (0, 1]")
    if not 0 < args.ood_target_recall <= 1:
        argparser.error("--ood_target_recall must be in (0, 1]")
    if args.ood_steps < 1:
        argparser.error("--ood_steps must be >= 1")
    if args.train and args.pretrained_checkpoint is None and args.checkpoint is None:
        argparser.error("--train needs --pretrained_checkpoint (unconditional) or --checkpoint (resume fine-tuning)")
    return args
