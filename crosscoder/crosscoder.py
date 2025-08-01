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
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_

# test
model = nnsight.LanguageModel("gpt2", device_map="auto")
config = model.config
model_config = config.to_dict() # type: ignore


cc_config = {
    "seed": 51,
    "batch_size": 2048, # number of activations processed in each training step
    "buffer_mult": 512, # multiplier for buffer size
    "lr": 4e-5, # learning rate for AdamW
    "num_tokens": int(4e8), # total number of tokens to process during the training run
    "l1_coefficient": 2.0, # weight for l1 sparsity reg (reduced from 2.0)
    "beta1": 0.9,
    "beta2": 0.999,
    "context": 1024, # context length for the model
    "device": "cuda",
    "model_batch_size": 32, # batch size when running the base model to generate activations
    "log_interval": 100,
    "save_interval": 100000,
    "model_name": "gpt2",
    "dtype": torch.float32,
    "ae_dim": 4096, # autoencoder dimension (increased from 1000)
    "drop_bos": True, # whether or not to drop the beginning of sentence token,
    "total_steps": 500000 # increased from 100000
}

SAVE_DIR = Path("./saves")


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
        if self.cfg["model_name"] == "gpt2":
            self.model = nnsight.LanguageModel("gpt2", device_map="auto")

        self.modelcfg = self.model.config.to_dict() # type: ignore
        self.num_layers = self.modelcfg['n_layer']
        self.resid_dim = self.modelcfg['n_embd']
        torch.manual_seed(42)
        self.ae_dim = cfg["ae_dim"]
        self.dtype = cfg["dtype"]
        self.save_dir = None

        self.W_enc = nn.Parameter(
            torch.empty(
                self.num_layers, self.resid_dim, self.ae_dim, dtype=self.dtype
            )
        )

        self.W_dec = nn.Parameter(
            torch.empty(
                self.ae_dim, self.num_layers, self.resid_dim, dtype=self.dtype
            )
        )

        nn.init.kaiming_uniform_(self.W_enc, a=1)
        nn.init.kaiming_uniform_(self.W_dec, a=1)

        dec_init_norm = 0.005
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * dec_init_norm
        )

        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "ae_dim n_layers d_model -> n_layers d_model ae_dim",
        )

        self.b_enc = nn.Parameter(torch.zeros(self.ae_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((self.num_layers, self.resid_dim), dtype=self.dtype)
        )

        # Move to device
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
        # x = batch n_layers d_model
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
            self.save_dir.mkdir(parents=True)

        # Add the missing save_version attribute
        self.save_version = getattr(self, 'save_version', 0)

        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        # Actually save the weights and config
        torch.save({
            'W_enc': self.W_enc.data,
            'W_dec': self.W_dec.data,
            'b_enc': self.b_enc.data,
            'b_dec': self.b_dec.data,
            'cfg': self.cfg
        }, weight_path)

        # Make config JSON serializable
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
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg["model_name"] == "gpt2":
            self.model = nnsight.LanguageModel("gpt2", device_map="auto")
        self.modelcfg = self.model.config.to_dict() # type: ignore
        self.buffer_size = self.cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["context"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["context"] - 1)

        self.buffer = torch.zeros(
            (self.buffer_size, self.modelcfg["n_layer"], self.modelcfg["n_embd"]),
            dtype = torch.float32,
            requires_grad=False,
        ).to(cfg["device"])
        self.pointer = 0
        self.first = True

        # Dynamic normalization will be computed per batch
        self.layer_means = None
        self.layer_stds = None
        self.normalization_computed = False

        self.refresh()

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer...")

        tokens = self.get_tokens_batch()

        all_acts = []

        for i in range(0, len(tokens), self.cfg["model_batch_size"]):
            batch_tokens = tokens[i:i + self.cfg["model_batch_size"]]

            with self.model.trace(batch_tokens) as tracer:

                layer_outputs = []
                for layer_idx in range(self.modelcfg["n_layer"]):

                    layer_out = self.model.transformer.h[layer_idx].output[0].save()
                    layer_outputs.append(layer_out)

            batch_acts = torch.stack(layer_outputs, dim=2)

            if self.cfg.get("drop_bos", True):
                batch_acts = batch_acts[:, 1:, :, :]

            batch_acts = batch_acts.reshape(-1, self.modelcfg["n_layer"], self.modelcfg["n_embd"])

            all_acts.append(batch_acts)

        # Concatenate all activations
        self.buffer = torch.cat(all_acts, dim=0)

        # Shuffle for better training
        perm = torch.randperm(len(self.buffer))
        self.buffer = self.buffer[perm]

        self.pointer = 0

        # Reset normalization flag to recompute with new data
        self.normalization_computed = False

    @torch.no_grad()
    def next(self):
        if self.pointer + self.cfg["batch_size"] > len(self.buffer):
            self.refresh()

        # Get batch of activations
        batch = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]

        # Apply dynamic normalization
        batch = batch.float()

        # Compute normalization statistics if not already done or periodically recompute
        if not self.normalization_computed or self.pointer % (self.cfg["batch_size"] * 100) == 0:
            # Use a larger sample for more stable statistics
            sample_size = min(self.cfg["batch_size"] * 10, len(self.buffer))
            sample_indices = torch.randperm(len(self.buffer))[:sample_size]
            sample_batch = self.buffer[sample_indices]

            self.layer_means = sample_batch.mean(dim=0, keepdim=True)  # Shape: (1, n_layers, d_model)
            self.layer_stds = sample_batch.std(dim=0, keepdim=True)    # Shape: (1, n_layers, d_model)
            self.normalization_computed = True
            print(f"Computed normalization stats - means: {self.layer_means.mean():.4f}, stds: {self.layer_stds.mean():.4f}")

        # Apply normalization
        batch = (batch - self.layer_means) / (self.layer_stds + 1e-8)

        return batch.to(self.cfg["device"])

    def get_tokens_batch(self):
        """Get a batch of tokenized text data"""
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

        tokens = []
        count = 0
        max_samples = self.buffer_batches

        for item in dataset:
            if count >= max_samples:
                break

            text = item['text']

            if len(text.strip()) < 50:
                continue

            token_ids = self.model.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.cfg["context"],
                truncation=True,
                padding="max_length"
            )

            tokens.append(token_ids)
            count += 1

        return torch.cat(tokens, dim=0)



class Trainer:
    def __init__(self, cfg, use_wandb=True):
        self.cfg = cfg
        self.model = model
        self.crosscoder = Crosscoder(cfg)
        self.buffer = Buffer(cfg)
        self.total_steps = cfg["total_steps"]
        self.use_wandb = use_wandb


        self.optimizer = torch.optim.AdamW(
        self.crosscoder.parameters(), lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

        self.step_counter: int = 0


        if use_wandb:
            wandb.init(project="crosscroders", entity="rohan-kathuria-neu", config=cfg)

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coefficient"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coefficient"]

    def step(self):
        acts = self.buffer.next()
        # Ensure acts are on the right device and dtype
        acts = acts.float().to(self.cfg["device"])

        # Get encoded activations for sparsity calculation
        with torch.no_grad():
            encoded_acts = self.crosscoder.encode(acts)

        losses = self.crosscoder.return_loss(acts)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()

        # Improved gradient clipping with monitoring
        grad_norm = clip_grad_norm_(self.crosscoder.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Enhanced loss monitoring
        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm.item(),
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

                # Periodic feature analysis
                if i % (self.cfg["log_interval"] * 10) == 0 and i > 0:
                    print(f"Analyzing feature quality at step {i}")
                    try:
                        analysis = self.analyze_feature_quality()
                        print(f"  Mean sparsity: {analysis['mean_sparsity']:.3f}")
                        print(f"  Dead features: {analysis['dead_features']}/{analysis['total_features']}")
                        print(f"  Worst layer reconstruction error: {max(analysis['layer_reconstruction_errors']):.4f}")

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

    def analyze_feature_quality(self, n_samples=1000):
        """Analyze the quality of learned features during training"""
        # Get a sample of activations
        sample_acts = self.buffer.next()[:n_samples]

        with torch.no_grad():
            # Encode the activations
            encoded = self.crosscoder.encode(sample_acts)

            # Compute feature statistics
            feature_means = encoded.mean(dim=0)
            feature_stds = encoded.std(dim=0)
            feature_sparsity = (encoded > 0).float().mean(dim=0)

            # Find most and least active features
            most_active = torch.argsort(feature_sparsity, descending=True)[:10]
            least_active = torch.argsort(feature_sparsity, descending=False)[:10]

            # Compute reconstruction quality
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
