from configs.default_cm_3_config import create_default_cm_3_config
from diffusion import DiffusionRunner

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = create_default_cm_3_config()
diffusion = DiffusionRunner(config)

diffusion.train()
