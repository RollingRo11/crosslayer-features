import torch
import torch.nn as nn
import torch.nn.functional as F
import nnsight
from nnsight import LanguageModel
import wandb
import einops
from typing import NamedTuple
from pathlib import Path
import tqdm
import json
import numpy as np
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from sophia import SophiaG
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent / "sae_vis" / "sae_vis"))
from model_utils import get_layer_output

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

cc_config = {
    "seed": 51,
    "batch_size": 2048,
    "buffer_mult": 16,
    "lr": 3e-5,
    "num_tokens": int(4e8),
    "l1_coefficient": 2.5,
    "beta1": 0.9,
    "beta2": 0.999,
    "context": 128,
    "device": "cuda",
    "model_batch_size": 16,
    "log_interval": 100,
    "save_interval": 50000,
    "model_name": "pythia",
    "dtype": torch.bfloat16,
    "ae_dim": 2**15,
    "drop_bos": True,
    "total_steps": 100000,
    "normalization": "layer_wise",
    "optimizer": "adamw", # Options: "adamw", "sophia"
    "dec_init_norm": 0.05,
}

# Use absolute path relative to this file's location
CROSSCODER_DIR = Path(__file__).parent
SAVE_DIR = CROSSCODER_DIR / "saves"
WANDB_DIR = CROSSCODER_DIR / "wandb"


# enc of n_layers, d_model, d_sae
# dec of d_sae, n_layers, d_model
# loss func (we gonna use L1 of norms)
# d_sae could be anyting
class LossOutput(NamedTuple):
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor

class Crosscoder(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.modelcfg = self.model.config.to_dict() # type: ignore
        if self.cfg["model_name"] == "gpt2":
            self.context = 1024
            self.num_layers = self.modelcfg['n_layer']
            self.resid_dim = self.modelcfg['n_embd']
        elif self.cfg["model_name"] == "pythia":
            self.context = 2048
            self.num_layers = self.modelcfg['num_hidden_layers']
            self.resid_dim = self.modelcfg['hidden_size']

        self.init_norm = cfg['dec_init_norm']

        self.seed = self.cfg["seed"]
        torch.manual_seed(self.seed)
        self.ae_dim = cfg["ae_dim"]
        self.dtype = cfg["dtype"]
        self.save_dir = None

        # [layers, model resid dim (embd), crosscoder blowup dim]
        self.W_enc = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.num_layers, self.resid_dim, self.ae_dim, dtype=self.dtype
                )
            )
        )

        # [crosscoder dim, layers, model resid dim (embd)]
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.ae_dim, self.num_layers, self.resid_dim, dtype=self.dtype
                )
            )
        )

        dec_init_norm = self.init_norm
        self.W_dec.data = self.W_dec.data * (dec_init_norm / self.W_dec.data.norm())


        enc_init_norm = self.init_norm
        self.W_enc.data = self.W_enc.data * (enc_init_norm / self.W_enc.data.norm())

        self.b_enc = nn.Parameter(torch.zeros(self.ae_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((self.num_layers, self.resid_dim), dtype=self.dtype)
        )

        self.to(cfg["device"])

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
            "... ae_dim, ae_dim n_layers d_model -> ... n_layers d_model"
        )

        return acts_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    def return_loss(self, x):
        x = x.to(self.dtype).to(self.cfg["device"])
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

    def save(self):
        if self.save_dir is None:
            # Ensure save directory exists
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            version_list = [
                int(file.name.split("_")[1])
                for file in list(SAVE_DIR.iterdir())
                if "version" in str(file)
            ]
            if len(version_list):
                version = 1 + max(version_list)
            else:
                version = 0
            self.save_dir = SAVE_DIR / f"version_{version}"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_version = getattr(self, 'save_version', 0)

        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save({
            'W_enc': self.W_enc.data,
            'W_dec': self.W_dec.data,
            'b_enc': self.b_enc.data,
            'b_dec': self.b_dec.data,
            'cfg': self.cfg
        }, weight_path)

        cfg_to_save = self.cfg.copy()
        cfg_to_save['dtype'] = str(cfg_to_save['dtype'])

        with open(cfg_path, 'w') as f:
            json.dump(cfg_to_save, f, indent=2)

        self.save_version += 1



class Buffer:
    """
    Data buffer - stores acts across all layers that can be used to train autoencoder.
    Will run model to generate more when it gets halfway empty.
    """
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        if self.cfg["model_name"] == "gpt2":
            self.context = 1024
        elif self.cfg["model_name"] == "pythia":
            self.context = 2048

        self.modelcfg = self.model.config.to_dict() # type: ignore

        # Handle different model architectures
        if 'n_layer' in self.modelcfg:
            self.num_layers = self.modelcfg['n_layer']
            self.resid_dim = self.modelcfg['n_embd']
        elif 'num_hidden_layers' in self.modelcfg:
            self.num_layers = self.modelcfg['num_hidden_layers']
            self.resid_dim = self.modelcfg['hidden_size']
        else:
            raise ValueError(f"Unsupported model architecture. Config keys: {list(self.modelcfg.keys())}")

        self.buffer_size = self.cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (self.context - 1)
        self.buffer_size = self.buffer_batches * (self.context - 1)

        self.buffer = torch.zeros(
            (self.buffer_size, self.num_layers, self.resid_dim),
            dtype = torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])
        self.pointer = 0
        self.first = True
        self.normalize = True

        estimated_norm_scaling_factors = self.estimate_norm_scaling_factor(cfg["model_batch_size"])

        self.normalisation_factor = torch.tensor(
            estimated_norm_scaling_factors,
            device=cfg["device"],
            dtype=torch.float32,
        )

        self.dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-100BT', split='train', streaming=False)
        self.dataset_iter = iter(self.dataset)

        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, n_batches_for_norm_estimate: int = 100):
        norms_per_layer = [[] for _ in range(self.num_layers)]

        all_tokens = self.get_tokens_for_norm_estimation(n_batches_for_norm_estimate, batch_size)

        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = all_tokens[i * batch_size : (i + 1) * batch_size]

            all_acts = []

            for j in range(0, len(tokens), self.cfg["model_batch_size"]):
                batch_tokens = tokens[j:j + self.cfg["model_batch_size"]]

                with self.model.trace(batch_tokens) as tracer:
                    layer_outputs = []
                    for layer_idx in range(self.num_layers):
                        layer_out = get_layer_output(self.model, layer_idx, tracer)
                        layer_outputs.append(layer_out)

                batch_acts = torch.stack(layer_outputs, dim=2)

                if self.cfg.get("drop_bos", True):
                    batch_acts = batch_acts[:, 1:, :, :]

                batch_acts = batch_acts.reshape(-1, self.num_layers, self.resid_dim)
                all_acts.append(batch_acts)

            acts = torch.cat(all_acts, dim=0)
            layer_norms = acts.norm(dim=-1)
            for layer_idx in range(self.num_layers):
                norms_per_layer[layer_idx].append(layer_norms[:, layer_idx].mean().item())

        scaling_factors = []
        for layer_idx in range(self.num_layers):
            mean_norm = np.mean(norms_per_layer[layer_idx])
            scaling_factor = np.sqrt(self.resid_dim) / mean_norm
            scaling_factors.append(scaling_factor)

        return scaling_factors

    @torch.no_grad()
    def get_tokens_for_norm_estimation(self, n_batches, batch_size):
        """Get all tokens needed for norm estimation at once"""
        # Use the same non-streaming dataset that was already loaded
        norm_dataset = self.dataset

        tokens = []
        count = 0
        total_samples = n_batches * batch_size

        for item in norm_dataset:
            if count >= total_samples:
                break

            text = item['text']

            if len(text.strip()) < 50:
                continue

            token_ids = self.model.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.context,
                truncation=True,
                padding="max_length"
            )
            tokens.append(token_ids)
            count += 1
        return torch.cat(tokens, dim=0)

    @torch.no_grad()
    def refresh(self):
        tokens = self.get_tokens_batch()

        all_acts = []

        for i in range(0, len(tokens), self.cfg["model_batch_size"]):
            batch_tokens = tokens[i:i + self.cfg["model_batch_size"]]

            with self.model.trace(batch_tokens) as tracer:

                layer_outputs = []
                for layer_idx in range(self.num_layers):

                    layer_out = get_layer_output(self.model, layer_idx, tracer)
                    layer_outputs.append(layer_out)

            batch_acts = torch.stack(layer_outputs, dim=2)

            if self.cfg.get("drop_bos", True):
                batch_acts = batch_acts[:, 1:, :, :]

            batch_acts = batch_acts.reshape(-1, self.num_layers, self.resid_dim)

            all_acts.append(batch_acts)

        self.buffer = torch.cat(all_acts, dim=0)

        perm = torch.randperm(len(self.buffer))
        self.buffer = self.buffer[perm]

        self.pointer = 0

    @torch.no_grad()
    def next(self):
        if self.pointer + self.cfg["batch_size"] > len(self.buffer):
            self.refresh()

        batch = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]].float()
        self.pointer += self.cfg["batch_size"]

        if self.normalize:
            batch = batch * self.normalisation_factor[None, :, None]

        return batch.to(self.cfg["device"])

    def get_tokens_batch(self):
        """Get a batch of tokenized text data"""

        tokens = []
        count = 0
        max_samples = self.buffer_batches

        try:
            for item in self.dataset_iter:
                if count >= max_samples:
                    break

                text = item['text']

                if len(text.strip()) < 50:
                    continue

                token_ids = self.model.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=self.context,
                    truncation=True,
                    padding="max_length"
                )
                tokens.append(token_ids)
                count += 1
        except StopIteration:
            self.dataset_iter = iter(self.dataset)
            for item in self.dataset_iter:
                if count >= max_samples:
                    break

                text = item['text']

                if len(text.strip()) < 50:
                    continue

                token_ids = self.model.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=self.context,
                    truncation=True,
                    padding="max_length"
                )
                tokens.append(token_ids)
                count += 1
        return torch.cat(tokens, dim=0)

class Trainer:
    def __init__(self, cfg, use_wandb=True):
        self.cfg = cfg
        if self.cfg["model_name"] == "gpt2":
            self.model = nnsight.LanguageModel("gpt2", device_map="auto")
            self.context = 1024
        elif self.cfg["model_name"] == "pythia":
            self.model = nnsight.LanguageModel("EleutherAI/pythia-2.8b-deduped", device_map="auto")
            self.context = 2048

        self.crosscoder = Crosscoder(cfg, model=self.model)
        self.buffer = Buffer(cfg, model=self.model)
        self.total_steps = cfg["total_steps"]
        self.use_wandb = use_wandb

        if cfg.get("optimizer", "adamw") == "sophia":
            self.optimizer = SophiaG(self.crosscoder.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)
        else:
            self.optimizer = torch.optim.AdamW(
                self.crosscoder.parameters(), lr=cfg["lr"],
                betas=(cfg["beta1"], cfg["beta2"]),
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

        self.step_counter: int = 0


        if use_wandb:
            WANDB_DIR.mkdir(exist_ok=True)
            wandb.init(
                project="crosscroders",
                entity="rohan-kathuria-neu",
                config=cfg,
                dir=str(WANDB_DIR)
            )

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        if self.step_counter < 0.08 * self.total_steps:
            return self.cfg["l1_coefficient"] * self.step_counter / (0.08 * self.total_steps)
        else:
            return self.cfg["l1_coefficient"]


    def step(self):
        acts = self.buffer.next()
        acts = acts.to(dtype=self.cfg["dtype"], device=self.cfg["device"])

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            losses = self.crosscoder.return_loss(acts)
            loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss

        loss.backward()

        # manually clip bcs torch grad_clip_norm_ cooked gpu vram (this is probably user error)
        grad_norm = 0.0
        max_norm = 1.0
        try:
            total_norm = 0.0
            param_count = 0
            for param in self.crosscoder.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                grad_norm = total_norm

                if total_norm > max_norm:
                    clip_coef = max_norm / (total_norm + 1e-6)
                    for param in self.crosscoder.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(clip_coef)
        except Exception as e:
            print(f"Gradient clipping failed: {e}, continuing without clipping")
            grad_norm = 0.0

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            encoded_acts = self.crosscoder.encode(acts)

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "sparsity": (encoded_acts > 0).float().mean().item(),
            "reconstruction_mse": losses.l2_loss.item() / acts.numel()
        }

        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        if self.use_wandb:
            wandb.log(loss_dict, step=self.step_counter)

    def save(self):
        self.crosscoder.save()


    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps):
                loss_dict = self.step()
                if i % self.cfg["log_interval"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_interval"] == 0:
                    print(f"Saving checkpoint at step {i+1}")
                    self.save()

                if i % (self.cfg["log_interval"] * 10) == 0 and i > 0:
                    try:
                        analysis = self.analyze()
                        if self.use_wandb:
                            wandb.log({
                                'feature_analysis/mean_sparsity': analysis['mean_sparsity'],
                                'feature_analysis/sparsity_std': analysis['sparsity_std'],
                                'feature_analysis/dead_features': analysis['dead_features'],
                                'feature_analysis/max_layer_error': max(analysis['layer_reconstruction_errors'])
                            }, step=self.step_counter)
                    except Exception as e:
                        print(f"  Feature analysis failed: {e}")

        finally:
            self.save()

    def analyze(self, n_samples=1000):
        """Analyze the quality of learned features during training"""
        sample_acts = self.buffer.next()[:n_samples]
        sample_acts = sample_acts.to(dtype=self.cfg["dtype"], device=self.cfg["device"])

        with torch.no_grad():
            encoded = self.crosscoder.encode(sample_acts)

            feature_means = encoded.mean(dim=0)
            feature_stds = encoded.std(dim=0)
            feature_sparsity = (encoded > 0).float().mean(dim=0)

            most_active = torch.argsort(feature_sparsity, descending=True)[:10]
            least_active = torch.argsort(feature_sparsity, descending=False)[:10]

            reconstructed = self.crosscoder.decode(encoded)
            recon_error = F.mse_loss(reconstructed, sample_acts, reduction='none')
            layer_recon_error = recon_error.mean(dim=[0, 2])  # Average over batch and d_model

            analysis = {
                'mean_sparsity': feature_sparsity.mean().item(),
                'sparsity_std': feature_sparsity.std().item(),
                'most_active_features': most_active.tolist(),
                'least_active_features': least_active.tolist(),
                'layer_reconstruction_errors': layer_recon_error.tolist(),
                'dead_features': (feature_sparsity < 1e-6).sum().item(),
                'total_features': len(feature_sparsity)
            }

            return analysis
