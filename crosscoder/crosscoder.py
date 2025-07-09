import torch
import torch.nn as nn
import torch.nn.functional as F
import nnsight
from nnsight import LanguageModel
import wandb
import einops
from typing import NamedTuple

model = nnsight.LanguageModel("gpt2", device="cuda")
config = model.config
print(config)
model_config = config.to_dict() # type: ignore

"""
GPT2Config {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.53.0",
  "use_cache": true,
  "vocab_size": 50257
}
"""

cc_config = {
    "seed": 51,
    "batch_size": 2048,
    "buffer_mult": 512,
    "lr": 2e-5,
    "num_tokens": int(4e8),
    "l1_coefficient": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "context": 1024,
    "device": "mps",
    "model_batch_size": 32,
    "log_interval": 100,
    "save_interval": 100000,
    "model_name": "gpt2",
    "dtype": torch.float32,
    "ae_dim": 1000
}


# enc of n_layers, d_model, d_sae
# dec of d_sae, n_layers, d_model
# loss func (we gonna use L1 of norms)
# d_sae could be anyting
class LossOutput(NamedTuple):
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor

class Crosscoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg == "gpt2":
            self.model = nnsight.LanguageModel("gpt2", device="cuda")

        self.modelcfg = self.model.config.to_dict() # type: ignore
        self.num_layers = self.modelcfg['n_layer']
        self.resid_dim = self.modelcfg['n_embd']
        torch.manual_seed(42)
        self.ae_dim = cfg["ae_dim"]
        self.dtype = cfg["dtype"]

        self.W_enc = nn.Parameter(
            torch.empty(
                self.num_layers, self.resid_dim, self.ae_dim
            )
        )

        self.W_dec = nn.Parameter(
            torch.empty(
                self.ae_dim, self.num_layers, self.resid_dim
            )
        )

        self.b_enc = nn.Parameter(torch.zeros(self.ae_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((self.num_layers, self.resid_dim), dtype=self.dtype)
        )

    def encode(self, x, apply_act=True):
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "... n_layers d_model, n_layers d_model ae_dim -> ... ae_dim"
        )

        if apply_act:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc

        return acts

    def decode(self, acts):
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "ae_dim, ae_dim n_layers d_model -> ... n_layers d_model"
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    def return_loss(self, x):
        # x = batch n_layers d_model
        x = x.to(self.dtype)
        acts = self.encode(x)

        reconstructed_x = self.decode(acts)
        squared_diff = (reconstructed_x.float() - x.float()).pow(2)
        l2_loss = (einops.reduce(squared_diff, "... n_layers d_model -> ...", "sum")).mean()

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: d_hidden n_layers
        total_decoder_norm = einops.reduce(
            decoder_norms, "d_hidden n_layers -> d_hidden", "sum"
        )

        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_loss = (acts > 0).float().sum(-1).mean()

        return LossOutput(l2_loss=l2_loss, l1_loss=l1_loss, l0_loss=l0_loss)

class Buffer:
    """
    Data buffer - stores acts across all layers that can be used to train autoencoder.
    Will run model to generate more when it gets halfway empty.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg["model_name"] == "gpt2":
            self.model = nnsight.LanguageModel("gpt2", device="cuda")
        self.modelcfg = self.model.config.to_dict() # type: ignore
        self.buffer_size = self.modelcfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["context"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)

        self.buffer = torch.zeros(
            (self.buffer_size, self.modelcfg["n_layer"], self.modelcfg["n_embd"]),
            dtype = torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])



class Trainer:
    def __init__(self, cfg, use_wandb=True):
        self.cfg = cfg
        self.model = model
        self.crosscoder = Crosscoder(cfg)
        self.buffer = Buffer(cfg)
        self.total_steps =

        self.optimizer = torch.optim.AdamW(
        self.crosscoder.parameters(), lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

        self.step_counter = 0

        if use_wandb:
            wandb.init(project="crosscroders", entity="rohan-kathuria-neu", config=cfg)

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def step(self):
        acts = self.buffer.next()
