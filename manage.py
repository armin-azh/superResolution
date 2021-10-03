from argparse import (Namespace,
                      ArgumentParser)
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from core.trainer import Trainer
from core.dataloader import ImageDataset
from settings import output_dir


def main(arguments: Namespace):
    if arguments.train_dir != "" and arguments.test_dir != "":
        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        save_path = output_dir.joinpath(_cu)
        save_path.mkdir(parents=True, exist_ok=True)

        print("Start training images")
        train_dir = Path(arguments.train_dir)
        test_dir = Path(arguments.test_dir)

        train_dataset = ImageDataset(input_dir=train_dir)
        test_dataset = ImageDataset(input_dir=test_dir)

        print(f"[Train] train: {len(train_dataset)} samples, test: {len(test_dataset)} samples")

        train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, num_workers=arguments.n_workers)
        test_loader = DataLoader(test_dataset, batch_size=arguments.batch_size, num_workers=arguments.n_workers)

        trainer = Trainer(lr=arguments.lr)
        trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=arguments.epochs, save_path=save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    # start options
    parser.add_argument('--train_dir', help="training directory", type=str, default="")
    parser.add_argument('--test_dir', help="test directory", type=str, default="")
    parser.add_argument('--model_dir', help="model state dictionary (.pth)", type=str, default="")

    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", help="batch size", type=int, default=2)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("--n_workers", help="number of threads", type=int, default=2)

    # end options
    args = parser.parse_args()
    main(arguments=args)
