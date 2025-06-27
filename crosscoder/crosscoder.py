import torch
import torch.nn as nn
import nnsight
from nnsight import LanguageModel

model = nnsight.LanguageModel("gpt2", device="cuda")
config = model.config
print(config)
config = config.to_dict() # type: ignore
print(config['n_embd'])


class Crosscoder(nn.Module):
    def __init__(self, cfg, ae_dim):
        self.num_layers = cfg['n_layer']
        self.resid_dim = cfg['n_embd']
        self.ae_dim = ae_dim
