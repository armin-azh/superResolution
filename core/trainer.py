from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np

from settings import has_cuda
from core.model import SuperResolutionModel


class Trainer:
    def __init__(self, lr: float):
        self._lr = lr

        self._model = SuperResolutionModel()

        if has_cuda:
            self._model.cuda()

        self._opt = optim.Adam(params=self._model.parameters(), lr=self._lr)

        self._criterion = nn.MSELoss()

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int, save_path: Path):

        total_loss = []
        total_valid_loss = []

        step = 0

        for epoch in range(epochs):
            # start training process
            tm_loss = []
            tm_valid_loss = []
            for idx, (x, y, z) in enumerate(train_loader):

                if has_cuda:
                    x = x.float().cuda()
                    y = y.float().cuda()
                    z = z.float().cuda()

                # start update
                self._opt.zero_grad()
                output_x, output_y = self._model(z)
                loss_x = self._criterion(output_x, x)
                loss_y = self._criterion(output_y, y)
                loss = loss_x + loss_y
                loss.backward()
                self._opt.step()
                # end update
                tm_loss.append(loss.item())
                step += 1

                if step % 100 == 0:
                    print(f"[{epoch+1}|epochs] | Train Loss: {tm_loss[-1]}")

            total_loss.append(np.array(tm_loss).mean())
            # end training process

            # start validation process
            for idx, (x, y, z) in enumerate(test_loader):

                if has_cuda:
                    x = x.float().cuda()
                    y = y.float().cuda()
                    z = z.float().cuda()

                with torch.no_grad():
                    output_x, output_y = self._model(z)
                    loss_x = self._criterion(output_x, x)
                    loss_y = self._criterion(output_y, y)
                    loss = loss_x + loss_y
                    tm_valid_loss.append(loss.item())

            total_valid_loss.append(np.array(tm_valid_loss).mean())

            print(f"[{epoch + 1}|epochs] | Train Loss: {total_loss[-1]}, Valid Loss: {total_valid_loss[-1]}")
            # end validation process


