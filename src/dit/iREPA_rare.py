from models.SiT_REPA_rare import DenoiserREPARare
from utils.util_rare import parse_args_iREPA_rare
from data.Dataloader_rare import get_rare_dataloader, get_rare_val_dataloader

if __name__ == "__main__":
    args = parse_args_iREPA_rare()

    if args.train:
        train_loader = get_rare_dataloader(
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            root=args.rare_root,
            centers=args.rare_centers.split(",") if args.rare_centers else None,
            balanced=args.rare_balanced,
        )
        model = DenoiserREPARare(args)
        model.load_for_finetuning()
        model.train_model(train_loader)

    elif args.sample:
        model = DenoiserREPARare(args)
        model.load_checkpoint()
        model.sample()

    elif args.ood:
        val_loader = get_rare_val_dataloader(
            batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers, root=args.rare_val_root
        )
        model = DenoiserREPARare(args)
        model.load_checkpoint()
        model.ood_evaluate(val_loader)
