from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from settings import *


class ImageDataset(Dataset):
    def __init__(self, input_dir: Path):
        self._ims_path = list(input_dir.glob("*"))

    def __len__(self):
        return len(self._ims_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _path = str(self._ims_path[idx])

        _img = Image.open(_path)
        _img = _img.convert('RGB')

        _img1 = _img.resize(image_dim_stage1)
        _img2 = _img.resize(image_dim_stage2)
        _img3 = _img.resize(image_dim_stage3)

        transformer = ToTensor()
        _img1 = transformer(_img1)
        _img2 = transformer(_img2)
        _img3 = transformer(_img3)
        return _img3, _img2, _img1

#
# if __name__ == '__main__':
#     im_path = Path("/home/lezarus/PycharmProjects/superResolution/data/dataset/1500 images")
#     dataset = ImageDataset(input_dir=im_path)
#     print(len(dataset))
#
#     print(dataset[1])
