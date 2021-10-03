from typing import List
import torch
from torch import (Tensor,
                   nn)
import torchsummary


class SuperResolutionModel(nn.Module):
    def __init__(self):
        super(SuperResolutionModel, self).__init__()

        # first block
        self._up_sample = nn.Upsample(scale_factor=(3, 3))
        self._conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=(9, 9))
        self._conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5))
        self._conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5))

        # second bock
        self._conv4 = nn.ConvTranspose2d(in_channels=9, out_channels=64, kernel_size=(9, 9))
        self._conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5))
        self._conv6 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5))

        self._relu = nn.ReLU(inplace=True)

    def forward(self, img: Tensor):
        """
        forward propagation
        :param img: tensor in shape (None, 3, h, w)
        :return:
        """
        x = self._up_sample(img)
        x = self._conv1(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._relu(x)
        x = self._conv3(x)
        x = self._relu(x)
        output = x
        bi_cub = nn.Upsample(size=output.shape[-2:], mode="bicubic", align_corners=True)
        nearest = nn.Upsample(size=output.shape[-2:])

        bi_img = bi_cub(img)
        nearest_img = nearest(img)

        concat = torch.cat([output, bi_img, nearest_img], dim=1)

        x = self._conv4(concat)
        x = self._relu(x)
        x = self._conv5(x)
        x = self._relu(x)
        x = self._conv6(x)
        return x, output


# if __name__ == '__main__':
#     model = SuperResolutionModel()
#
#     torchsummary.summary(model=model, input_data=(3, 200, 300))
