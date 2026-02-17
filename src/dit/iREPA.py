from models.SiT_REPA import DenoiserREPA
from utils.util import parse_args_iREPA
from data.Dataloader import get_gastronet_dataloader, get_gastronet_full_dataloader

if __name__ == "__main__":
    args = parse_args_iREPA()

    if args.train:
        if args.full:
            print("Training on full dataset...")
            train_loader = get_gastronet_full_dataloader(
                batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers,folder=args.folder
            )
        else:
            train_loader = get_gastronet_dataloader(
                batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers
            )
        model = DenoiserREPA(args)
        model.train_model(train_loader)

    elif args.sample:
        model = DenoiserREPA(args)
        model.load_checkpoint()
        model.sample()

    elif args.fid:
        model = DenoiserREPA(args)
        model.load_checkpoint()
        model.fid_sample()

    elif args.inpaint:
        model = DenoiserREPA(args)
        model.load_checkpoint()
        train_loader = get_gastronet_dataloader(
            batch_size=args.num_samples, img_size=args.img_size, num_workers=args.num_workers
        )
        model.inpaint(train_loader)

    elif args.outpaint:
        model = DenoiserREPA(args)
        model.load_checkpoint()
        train_loader = get_gastronet_dataloader(
            batch_size=args.num_samples, img_size=args.img_size, num_workers=args.num_workers
        )
        model.outpaint(train_loader)
