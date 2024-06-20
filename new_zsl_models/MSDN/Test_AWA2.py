import torch
from MSDN import MSDN
from dataset import UNIDataloader
import argparse
import json
from utils import evaluation


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/test_AWA2.json')
config = parser.parse_args()
with open(config.config, 'r') as f:
    config.__dict__ = json.load(f)

dataloader = UNIDataloader(config)

model_gzsl = MSDN(config, normalize_V=True, normalize_F=True, is_conservative=True,
                  uniform_att_1=False, uniform_att_2=False,
                  is_conv=False, is_bias=True).to(config.device)
model_dict = model_gzsl.state_dict()
saved_dict = torch.load('saved_model/AWA2_MSDN_GZSL.pth')
saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
model_dict.update(saved_dict)
model_gzsl.load_state_dict(model_dict)

model_czsl = MSDN(config, normalize_V=True, normalize_F=True, is_conservative=True,
                  uniform_att_1=False, uniform_att_2=False,
                  is_conv=False, is_bias=True).to(config.device)
model_dict = model_czsl.state_dict()
saved_dict = torch.load('saved_model/AWA2_MSDN_CZSL.pth')
saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
model_dict.update(saved_dict)
model_czsl.load_state_dict(model_dict)

evaluation(config.batch_size, config.device,
           dataloader, model_gzsl, model_czsl)
