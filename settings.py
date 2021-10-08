import os
from pathlib import Path

import torch

project_dir = Path(os.path.dirname(os.path.abspath(__file__)))

output_dir = project_dir.joinpath("result")
output_dir.mkdir(parents=True, exist_ok=True)

has_cuda = torch.cuda.is_available()

image_dim_stage1 = (200, 100)
image_dim_stage2 = (216, 116)
image_dim_stage3 = (232, 132)
