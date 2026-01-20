from models.SiT_REPA import DenoiserREPA
from utils.util import parse_args_iREPA
from data.Dataloader import get_gastronet_dataloader

if __name__ == "__main__":

    args = parse_args_iREPA()

    if args.train:

        train_loader = get_gastronet_dataloader(batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
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

    elif args.feature_extractor:
        model = DenoiserREPA(args)
        model.load_checkpoint()
        # obviously this is just a placeholder for actual data
        train_loader = get_gastronet_dataloader(batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
        model.feature_extractor(train_loader)