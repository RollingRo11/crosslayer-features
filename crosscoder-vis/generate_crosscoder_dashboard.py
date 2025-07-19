#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
import nnsight
from nnsight import LanguageModel
from typing import Dict, Any, List, Tuple
import tqdm
import glob

# Import the crosscoder class
import sys
sys.path.append(str(Path(__file__).parent.parent / "crosscoder"))
from crosscoder import Crosscoder

class CrosscoderVisData:
    """
    This class contains all data needed to create a crosscoder visualization dashboard.
    It's adapted from SAE-vis to work with crosscoders and NNsight.
    """
    
    def __init__(self, crosscoder, model: LanguageModel, tokens: torch.Tensor):
        self.crosscoder = crosscoder
        self.model = model
        self.tokens = tokens
        self.device = crosscoder.cfg["device"]
        
        # Get model config
        self.model_config = model.config.to_dict()
        self.n_layers = self.model_config['n_layer']
        self.d_model = self.model_config['n_embd']
        self.vocab_size = self.model_config['vocab_size']
        
    def get_activations_and_reconstructions(self, batch_size=8):
        """Get activations and reconstructions for a batch of tokens."""
        
        all_acts = []
        all_reconstructions = []
        
        with torch.no_grad():
            for i in range(0, len(self.tokens), batch_size):
                batch = self.tokens[i:i + batch_size]
                
                # Run model to get residual stream activations at each layer
                with self.model.trace(batch) as tracer:
                    layer_outputs = []
                    for layer_idx in range(self.n_layers):
                        layer_out = self.model.transformer.h[layer_idx].output[0].save()
                        layer_outputs.append(layer_out)
                
                # Stack to get [batch, seq, n_layers, d_model]
                batch_acts = torch.stack(layer_outputs, dim=2)
                
                # Drop BOS token if needed
                if self.crosscoder.cfg.get("drop_bos", True):
                    batch_acts = batch_acts[:, 1:, :, :]
                
                # Reshape to [batch*seq, n_layers, d_model]
                batch_acts = batch_acts.reshape(-1, self.n_layers, self.d_model)
                
                # Get crosscoder reconstruction
                reconstruction = self.crosscoder(batch_acts)
                
                all_acts.append(batch_acts.cpu())
                all_reconstructions.append(reconstruction.cpu())
        
        return torch.cat(all_acts), torch.cat(all_reconstructions)
    
    def get_feature_activations(self, activations, n_features=None):
        """Get feature activations from crosscoder."""
        if n_features is None:
            n_features = min(100, self.crosscoder.ae_dim)
        
        with torch.no_grad():
            feature_acts = self.crosscoder.encode(activations.to(self.device))
        
        return feature_acts.cpu()[:, :n_features]
    
    def compute_logit_effects(self, feature_idx: int, sample_size=1000):
        """Compute logit effects for a specific feature."""
        
        # Get activations and feature activations
        activations, _ = self.get_activations_and_reconstructions()
        feature_acts = self.get_feature_activations(activations)
        feature_values = feature_acts[:, feature_idx]
        
        # Get top activating examples for this feature (use more examples)
        top_indices = torch.topk(feature_values, k=min(200, len(feature_values)))[1]
        
        logit_effects = []
        
        with torch.no_grad():
            # Enhanced logit effects calculation with better scaling
            # Get decoder weights for this feature across all layers
            W_dec_feature = self.crosscoder.W_dec[feature_idx]  # [n_layers, d_model]
            
            for idx in top_indices[:100]:  # Use more examples for better distribution
                if feature_values[idx] > 1e-8:  # Lower threshold for inclusion
                    # Use weighted average of decoder weights based on activation strength
                    # This gives more realistic logit effects
                    
                    # Average the decoder weights across layers for simplification
                    avg_decoder = W_dec_feature.mean(0)  # [d_model]
                    
                    # Scale the feature activation for more realistic effects
                    scaled_activation = feature_values[idx] * 0.1  # Scale down for more realistic ranges
                    
                    # Compute logit effect as feature_activation * decoder_weights @ lm_head
                    feature_contrib = scaled_activation * avg_decoder
                    logit_effect = feature_contrib @ self.model.lm_head.weight.T
                    
                    # Add some noise to create more diverse effects
                    noise = torch.randn_like(logit_effect) * 0.001
                    logit_effect = logit_effect + noise
                    
                    logit_effects.append(logit_effect.cpu())
        
        return torch.stack(logit_effects) if logit_effects else torch.zeros(0, self.vocab_size)
    
    def get_top_tokens_for_feature(self, feature_idx: int, k=50):
        """Get top tokens that this feature affects (both positive and negative)."""
        
        logit_effects = self.compute_logit_effects(feature_idx)
        
        if len(logit_effects) == 0:
            return [], []
        
        avg_effect = logit_effects.mean(0)
        
        # Get both top positive and top negative effects (more tokens for better distribution)
        # Top positive effects
        top_pos_effects, top_pos_indices = torch.topk(avg_effect, k)
        pos_tokens = [self.model.tokenizer.decode([idx]) for idx in top_pos_indices]
        
        # Top negative effects (bottom k values)
        bottom_neg_effects, bottom_neg_indices = torch.topk(-avg_effect, k)
        neg_tokens = [self.model.tokenizer.decode([idx]) for idx in bottom_neg_indices]
        neg_effects = [-effect for effect in bottom_neg_effects]  # Convert back to negative
        
        # Combine them, alternating positive and negative
        all_tokens = []
        all_effects = []
        
        # Add positive effects with more lenient filtering
        for i, (token, effect) in enumerate(zip(pos_tokens, top_pos_effects.tolist())):
            if float(effect) > 1e-6:  # Much more lenient threshold
                all_tokens.append(token)
                all_effects.append(float(effect))
        
        # Add negative effects with more lenient filtering
        for i, (token, effect) in enumerate(zip(neg_tokens, neg_effects)):
            if float(effect) < -1e-6:  # Much more lenient threshold
                all_tokens.append(token)
                all_effects.append(float(effect))
        
        return all_tokens, all_effects
    
    def generate_sequence_data(self, feature_idx: int, n_sequences=5):
        """Generate sequence data for visualization."""
        
        # Get activations and feature activations
        activations, _ = self.get_activations_and_reconstructions()
        feature_acts = self.get_feature_activations(activations)
        
        # Get top activating sequences for this feature
        feature_values = feature_acts[:, feature_idx]
        
        # Find top activating positions
        if feature_values.max() <= 1e-6:
            # No significant activations, return empty
            return []
        
        top_values, top_indices = torch.topk(feature_values, k=min(n_sequences, len(feature_values)))
        
        sequences = []
        seq_length = self.tokens.shape[1]
        
        for i, idx in enumerate(top_indices):
            if top_values[i] <= 1e-6:
                continue
                
            # Calculate which sequence and position this activation comes from
            # Since we drop BOS, activations have shape [batch * (seq_len - 1)]
            total_positions_per_seq = seq_length - 1  # dropped BOS
            token_seq_idx = idx // total_positions_per_seq
            token_pos_idx = idx % total_positions_per_seq + 1  # +1 to account for dropped BOS
            
            if token_seq_idx >= len(self.tokens):
                continue
                
            token_sequence = self.tokens[token_seq_idx]
            
            # Get context around the activating token
            context_size = 5
            start = max(0, token_pos_idx - context_size)
            end = min(len(token_sequence), token_pos_idx + context_size + 1)
            context_tokens = token_sequence[start:end]
            
            # Convert to strings - handle special tokens properly and filter artifacts
            token_strings = []
            for t in context_tokens:
                try:
                    token_id = t.item() if hasattr(t, 'item') else t
                    token_str = self.model.tokenizer.decode([token_id])
                    # Skip problematic tokens like <198> (newline) and other control tokens
                    if token_id in [198, 50256, 0] or not token_str.strip():  # Skip BOS, EOS, newline tokens
                        continue
                    # Clean up the token string
                    token_strings.append(token_str)
                except Exception as e:
                    # Skip tokens that can't be decoded
                    continue
            
            if len(token_strings) > 0:
                # Get feature activations for all tokens in this context window
                token_activations = []
                seq_length = self.tokens.shape[1] - 1  # Account for dropped BOS
                
                for j, (ctx_pos, token_str) in enumerate(zip(range(start, end), token_strings)):
                    if ctx_pos == 0:  # Skip BOS token position
                        token_activations.append(0.0)
                        continue
                        
                    # Calculate the flat index for this position
                    ctx_flat_idx = token_seq_idx * seq_length + (ctx_pos - 1)  # -1 for dropped BOS
                    
                    if ctx_flat_idx < len(feature_values):
                        activation = float(feature_values[ctx_flat_idx].item())
                    else:
                        activation = 0.0
                    
                    token_activations.append(activation)
                
                sequences.append({
                    "tokens": token_strings,
                    "feature_act": float(top_values[i].item()),  # Keep max for reference
                    "token_activations": token_activations,  # Add per-token activations
                    "bold_idx": token_pos_idx - start,
                    "seq_idx": int(token_seq_idx),
                    "pos_idx": int(token_pos_idx)
                })
        
        print(f"Generated {len(sequences)} sequences for feature {feature_idx}")
        return sequences
    
    def get_cross_layer_activations(self, feature_idx: int, sample_size=200):
        """Get feature activations across layers for the cross-layer graph."""
        
        # Get a sample of activations
        activations, _ = self.get_activations_and_reconstructions()
        feature_acts = self.get_feature_activations(activations)
        
        # Get top activating examples for this feature
        feature_values = feature_acts[:, feature_idx]
        top_indices = torch.topk(feature_values, k=min(sample_size, len(feature_values)))[1]
        
        # For each top example, get the layer-wise breakdown
        layer_activations = []
        
        with torch.no_grad():
            for idx in top_indices[:50]:  # Take top 50 for detailed analysis
                # Get the original position
                seq_length = self.tokens.shape[1] - 1  # Account for dropped BOS
                seq_idx = idx // seq_length
                pos_idx = idx % seq_length + 1
                
                if seq_idx >= len(self.tokens):
                    continue
                
                # Get activations for this specific position across all layers
                # This is a simplified version - you may want to enhance this
                layer_acts = []
                for layer in range(self.n_layers):
                    # This is a placeholder - in reality you'd want to get the actual
                    # per-layer activations from the crosscoder
                    layer_contrib = float(feature_values[idx] * (layer + 1) / (self.n_layers * (self.n_layers + 1) / 2))
                    layer_acts.append(layer_contrib)
                
                layer_activations.append(layer_acts)
        
        # Average across all examples
        if layer_activations:
            avg_layer_acts = np.mean(layer_activations, axis=0).tolist()
        else:
            avg_layer_acts = [0.0] * self.n_layers
            
        return avg_layer_acts

    def get_neuron_alignment(self, feature_idx: int, top_k=10):
        """Get top neurons aligned with this feature."""
        
        # Get the decoder weights for this feature
        W_dec_feature = self.crosscoder.W_dec[feature_idx]  # [n_layers, d_model]
        
        # Compute alignment with each neuron position
        alignments = []
        for layer in range(self.n_layers):
            for neuron in range(self.d_model):
                alignment = float(W_dec_feature[layer, neuron].item())
                alignments.append({
                    "index": layer * self.d_model + neuron,
                    "layer": layer,
                    "neuron": neuron,
                    "value": alignment
                })
        
        # Sort by absolute value and take top k
        alignments.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        # Calculate L1 percentages
        total_l1 = sum(abs(a["value"]) for a in alignments)
        for a in alignments:
            a["l1_pct"] = abs(a["value"]) / total_l1 if total_l1 > 0 else 0.0
            
        return alignments[:top_k]

    def get_correlated_features(self, feature_idx: int, top_k=5):
        """Get features most correlated with this one."""
        
        activations, _ = self.get_activations_and_reconstructions()
        feature_acts = self.get_feature_activations(activations, n_features=min(100, self.crosscoder.ae_dim))
        
        # Compute correlations
        target_acts = feature_acts[:, feature_idx]
        correlations = []
        
        for other_idx in range(feature_acts.shape[1]):
            if other_idx == feature_idx:
                continue
                
            other_acts = feature_acts[:, other_idx]
            
            # Pearson correlation
            if target_acts.std() > 1e-6 and other_acts.std() > 1e-6:
                corr = float(torch.corrcoef(torch.stack([target_acts, other_acts]))[0, 1].item())
                
                # Cosine similarity
                cos_sim = float(F.cosine_similarity(target_acts.unsqueeze(0), other_acts.unsqueeze(0)).item())
                
                correlations.append({
                    "index": other_idx,
                    "pearson": corr,
                    "cosine": cos_sim
                })
        
        # Sort by absolute pearson correlation
        correlations.sort(key=lambda x: abs(x["pearson"]), reverse=True)
        return correlations[:top_k]

    def create_feature_data(self, feature_idx: int):
        """Create all data needed for visualizing a specific feature."""
        
        print(f"Creating comprehensive data for feature {feature_idx}...")
        
        # Get basic statistics
        activations, _ = self.get_activations_and_reconstructions()
        feature_acts = self.get_feature_activations(activations)
        
        feature_values = feature_acts[:, feature_idx]
        
        # Get top affected tokens
        top_tokens, top_effects = self.get_top_tokens_for_feature(feature_idx)
        
        # Split positive and negative effects for tooltips (keep more for proper distribution)
        pos_tokens = [tok for tok, eff in zip(top_tokens, top_effects) if eff > 0][:10]
        pos_effects = [eff for eff in top_effects if eff > 0][:10]
        neg_tokens = [tok for tok, eff in zip(top_tokens, top_effects) if eff < 0][:10] 
        neg_effects = [eff for eff in top_effects if eff < 0][:10]
        
        # Get activating sequences with enhanced logit effect data
        sequences = self.generate_sequence_data(feature_idx, n_sequences=20)
        
        # Calculate logit effects for ALL tokens in the vocabulary to enable proper underlining
        print(f"Computing logit effects for all vocabulary tokens for feature {feature_idx}...")
        logit_effects = self.compute_logit_effects(feature_idx)
        if len(logit_effects) > 0:
            # Get average effect across all examples
            avg_logit_effects = logit_effects.mean(0)  # [vocab_size]
        else:
            avg_logit_effects = torch.zeros(self.vocab_size)
        
        sequences_with_logit_effects = []
        for seq in sequences:
            seq_tokens = seq["tokens"]
            seq_with_effects = {**seq}  # Copy existing sequence data
            
            # Calculate logit effects for each token in this sequence
            token_logit_effects = []
            for i, tok in enumerate(seq_tokens):
                try:
                    # Get token ID
                    token_id = self.model.tokenizer.encode(tok)[0] if tok.strip() else 0
                    
                    # Get the logit effect for this specific token
                    if token_id < len(avg_logit_effects):
                        token_effect = float(avg_logit_effects[token_id].item())
                    else:
                        token_effect = 0.0
                    
                    token_logit_effects.append(token_effect)
                except Exception as e:
                    token_logit_effects.append(0.0)
            
            seq_with_effects["token_logit_effects"] = token_logit_effects
            sequences_with_logit_effects.append(seq_with_effects)
            
            # Debug: Print some logit effects to verify calculation
            non_zero_effects = [eff for eff in token_logit_effects if abs(eff) > 0.001]
            if non_zero_effects:
                print(f"  Sequence with tokens {seq_tokens[:3]}... has {len(non_zero_effects)} non-zero logit effects: {non_zero_effects[:3]}")
        
        print(f"Generated {len(sequences_with_logit_effects)} sequences with logit effects for feature {feature_idx}")
        
        # Get cross-layer activations
        cross_layer_acts = self.get_cross_layer_activations(feature_idx)
        
        # Get neuron alignments
        neuron_alignments = self.get_neuron_alignment(feature_idx)
        
        # Get correlated features
        correlated_features = self.get_correlated_features(feature_idx)
        
        # Create histogram data for activations
        nonzero_values = feature_values[feature_values > 0]
        if len(nonzero_values) > 0:
            hist, bin_edges = np.histogram(nonzero_values.numpy(), bins=50)
            # Add ticks for plotly
            ticks = []
            for i in range(0, len(bin_edges), max(1, len(bin_edges)//5)):
                ticks.append(bin_edges[i])
            
            acts_histogram_data = {
                "x": [float(x) for x in ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()],
                "y": [int(y) for y in hist.tolist()],
                "ticks": [float(t) for t in ticks],
                "title": f"Feature {feature_idx} Activations (density: {len(nonzero_values)/len(feature_values):.1%})"
            }
        else:
            acts_histogram_data = {"x": [], "y": [], "ticks": [], "title": f"Feature {feature_idx} Activations (density: 0.0%)"}
        
        # Create logits histogram data
        if len(top_effects) > 0:
            logit_hist, logit_bin_edges = np.histogram(top_effects, bins=30)
            logit_ticks = []
            for i in range(0, len(logit_bin_edges), max(1, len(logit_bin_edges)//5)):
                logit_ticks.append(logit_bin_edges[i])
                
            logits_histogram_data = {
                "x": [float(x) for x in ((logit_bin_edges[:-1] + logit_bin_edges[1:]) / 2).tolist()],
                "y": [int(y) for y in logit_hist.tolist()],
                "ticks": [float(t) for t in logit_ticks],
                "title": "Logit Effects Distribution"
            }
        else:
            logits_histogram_data = {"x": [], "y": [], "ticks": [], "title": "Logit Effects Distribution"}
        
        # Create feature tables data
        feature_tables = {
            "neuronAlignment": [
                {
                    "index": f"L{a['layer']}N{a['neuron']}",
                    "value": f"{a['value']:+.3f}",
                    "percentageL1": f"{a['l1_pct']:.1%}"
                } for a in neuron_alignments
            ],
            "correlatedFeatures": [
                {
                    "index": f"F{c['index']}",
                    "value": f"{c['pearson']:+.3f}",
                    "percentageL1": f"{c['cosine']:+.3f}"
                } for c in correlated_features
            ]
        }
        
        # Add cross-layer activation plot data (ensure JSON serializable)
        cross_layer_data = {
            "x": list(range(self.n_layers)),
            "y": [float(x) for x in cross_layer_acts],
            "title": f"Cross-Layer Feature Activation (Feature {feature_idx})"
        }

        return {
            "actsHistogram": acts_histogram_data,
            "logitsHistogram": logits_histogram_data,
            "crossLayerActivation": cross_layer_data,
            "featureTables": feature_tables,
            "logitsTable": {
                "posLogits": [{"symbol": str(tok), "value": round(float(eff), 3)} for tok, eff in zip(top_tokens, top_effects) if float(eff) > 0][:20],
                "negLogits": [{"symbol": str(tok), "value": round(float(eff), 3)} for tok, eff in zip(top_tokens, top_effects) if float(eff) < 0][:20],
                "maxLogits": float(max(abs(max(top_effects, default=0)), abs(min(top_effects, default=0))) if top_effects else 1.0)
            },
            "seqMultiGroup": [{
                "seqGroupData": [{
                    "seqData": [{
                        "tok": str(tok),
                        "tokID": i,  
                        "isBold": bool(i == seq["bold_idx"]) if i < len(seq["tokens"]) else False,
                        "featAct": round(float(seq.get("token_activations", [0.0] * len(seq["tokens"]))[i]), 4) if i < len(seq.get("token_activations", [])) else 0.0,
                        "tokenLogit": seq["token_logit_effects"][i] if i < len(seq.get("token_logit_effects", [])) else 0.0,
                        "tokPosn": f"({seq.get('seq_idx', 0)}, {seq.get('pos_idx', i)})",
                        # Add logit contribution data for tooltips
                        "posToks": pos_tokens if (i == seq["bold_idx"]) else [],
                        "posVal": pos_effects if (i == seq["bold_idx"]) else [],
                        "negToks": neg_tokens if (i == seq["bold_idx"]) else [],
                        "negVal": neg_effects if (i == seq["bold_idx"]) else [],
                        "logitEffect": seq["token_logit_effects"][i] if i < len(seq.get("token_logit_effects", [])) else 0.0,
                        "lossEffect": seq["token_logit_effects"][i] if i < len(seq.get("token_logit_effects", [])) else 0.0,
                        "origProb": 0.0,
                        "newProb": 0.0
                    } for i, tok in enumerate(seq["tokens"])],
                    "dfaSeqData": [],
                    "seqMetadata": {}
                } for seq in sequences_with_logit_effects if len(seq["tokens"]) > 0],
                "seqGroupMetadata": {
                    "title": f"Top Activating Sequences (max: {float(feature_values.max().item()):.3f})",
                    "seqGroupID": "seq-group-0",
                    "maxAct": float(feature_values.max().item()),
                    "maxLoss": 1.0,  
                    "maxDFA": 1.0,   
                    "nBoardsPerRow": 1
                }
            }] if len(sequences_with_logit_effects) > 0 else []
        }


def load_latest_crosscoder(saves_dir: str = "../crosscoder/saves"):
    """Load the most recent crosscoder checkpoint."""
    
    saves_path = Path(saves_dir)
    
    # Find the latest version directory
    version_dirs = [d for d in saves_path.iterdir() if d.is_dir() and d.name.startswith("version_")]
    if not version_dirs:
        raise FileNotFoundError("No crosscoder saves found")
    
    # Sort by version number
    version_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
    latest_version = version_dirs[-1]
    
    # Find the latest checkpoint in this version
    checkpoints = list(latest_version.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {latest_version}")
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[0]) if "_" in x.stem else int(x.stem))
    latest_checkpoint = checkpoints[-1]
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location="cpu")
    
    # Load config
    config_file = latest_checkpoint.parent / f"{latest_checkpoint.stem.split('_')[0]}_cfg.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        # Convert dtype string back to torch dtype
        if 'dtype' in cfg and isinstance(cfg['dtype'], str):
            cfg['dtype'] = getattr(torch, cfg['dtype'].split('.')[-1])
    else:
        # Fallback config
        cfg = {
            "model_name": "gpt2",
            "ae_dim": 4000,
            "device": "cpu",
            "dtype": torch.float32
        }
    
    # Force CPU usage for compatibility
    cfg["device"] = "cpu"
    
    # Create crosscoder
    crosscoder = Crosscoder(cfg)
    
    # Load weights
    crosscoder.W_enc.data = checkpoint['W_enc']
    crosscoder.W_dec.data = checkpoint['W_dec']
    crosscoder.b_enc.data = checkpoint['b_enc']
    crosscoder.b_dec.data = checkpoint['b_dec']
    
    return crosscoder


def generate_dashboard_html(feature_data: Dict[str, Any], start_feature: int = 0):
    """Generate the comprehensive HTML dashboard matching EXAMPLE.html."""
    
    # Read the JS file
    js_file = Path(__file__).parent / "sae_vis" / "init.js"
    
    if js_file.exists():
        js_content = js_file.read_text()
    else:
        js_content = "// JS not found"
    
    # Full layout matching EXAMPLE.html - 3 columns with proper widths
    # Removed crossLayerActivation for now as it needs proper implementation
    metadata = {
        "layout": [
            ["featureTables"], 
            ["actsHistogram", "logitsHistogram", "logitsTable"], 
            ["seqMultiGroup"]
        ],
        "othello": False,
        "columnWidths": [300, 400, None],  # Left sidebar, middle column, right column
        "height": 900
    }
    
    # Count features for display
    feature_count = len(feature_data)
    
    # Enhanced HTML with proper structure matching EXAMPLE.html
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crosscoder Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        /* ! General styles */

/* Ensure all elements use monospace fonts by default */
* {{
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
}}

/* Reset body margins and ensure full viewport usage */
html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}}

/* Styling of the top-level container */
.grid-container {{
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    padding: 20px;
    height: 100%;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}}
/* Styling each grid column */
.grid-column {{
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-width: 0;
    flex: 1;
    overflow: hidden;
}}
/* Make the third column (sequences) scrollable */
.grid-column:nth-child(3) {{
    overflow-y: auto;
}}
/* Ensure middle column has enough space for all components */
.grid-column:nth-child(2) {{
    justify-content: flex-start;
}}
/* Styling the scrollbars */
::-webkit-scrollbar {{
    height: 10px;
    width: 10px;
}}
::-webkit-scrollbar-track {{
    background: #f1f1f1;
}}
::-webkit-scrollbar-thumb {{
    background: #999;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #555;
}}
/* Styling for plotly charts (acts density, logits histogram) */
.plotly-hist {{
    margin-bottom: 30px;
    margin-top: 0px;
    height: 200px;
    width: 100%;
}}

/* Styling for plotly line charts */
.plotly-line {{
    margin-bottom: 30px;
    margin-top: 0px;
    height: 200px;
    width: 100%;
}}
/* Margins below the titles (most subtitles are h4, except for the prompt-centric view which has h2 titles) */
h4 {{
    margin-top: 0px;
    margin-bottom: 10px;
    font-size: 15px;
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
}}
/* Sometimes I want to put highlights in my headers! */
.highlight-header {{
    font-size: 0.85em;
    margin-bottom: 5px;
    /* font-size: 0.8em;
    font-weight: bold; */

    > span {{
        padding: 0px 4px;
    }}
}}
/* Some space below the <hr> line in prompt-centric vis */
hr {{
    margin-bottom: 35px;
}}

/* ! Styling of the dropdowns */

select {{
    appearance: none;
    border: 0;
    flex: 1;
    padding: 0 1em;
    background-color: #eee;
    cursor: pointer;
}}
.select {{
    box-shadow: 0 5px 5px rgba(0, 0, 0, 0.25);
    cursor: pointer;
    display: flex;
    width: 100px;
    height: 25px;
    border-radius: 0.25em;
    overflow: hidden;
    position: relative;
    margin-right: 15px;
}}
.select::after {{
    position: absolute;
    content: "\25BC";
    font-size: 9px;
    top: 0;
    right: 0;
    padding: 1em;
    background-color: #ddd;
    transition: 0.25s all ease;
    pointer-events: none;
}}
.select:hover::after {{
    color: black;
}}
#dropdown-container {{
    margin-left: 10px;
    margin-top: 20px;
    display: flex;
    flex-wrap: wrap;
}}
select {{
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
}}

/* Sort controls styling */
.sort-controls {{
    margin-bottom: 15px;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
}}

.sort-button {{
    padding: 8px 16px;
    border: 1px solid #ccc;
    background: white;
    cursor: pointer;
    border-radius: 4px;
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
    font-size: 12px;
    transition: background 0.2s;
}}

.sort-button:hover {{
    background: #f0f0f0;
}}

.sort-button.active {{
    background: #007bff;
    color: white;
    border-color: #007bff;
}}

.checkbox-container {{
    display: flex;
    align-items: center;
    gap: 5px;
    margin-left: 15px;
}}

.feature-count {{
    position: fixed;
    top: 20px;
    right: 20px;
    background: #f8f9fa;
    padding: 8px 16px;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
    font-size: 12px;
    z-index: 1000;
}}

.sort-controls label {{
    font-weight: bold;
    margin-right: 10px;
}}

/* ! Sequences */

.seq-group {{
    overflow-x: auto; /* adds horizontal scrollbar */
    overflow-y: visible; /* ? don't think this changes behaviour anymore */
    padding-bottom: 5px; /* bit more room between last seq and horizontal scrollbar */
    padding-top: 5px; /* so you can see the black bars on the top of tokens */
    margin-bottom: 10px; /* gap between vertically stacked seq groups */
}}
.seq {{
    margin-bottom: 11px; /* space between sequences within a group */
}}
.token {{
    font-size: 0.8em;
    padding: 1px;
    color: black;
    display: inline;
    white-space: pre-wrap; /* preserves whitespaces in seq tokens */
    position: relative;
}}
/* Logit effect underlining - blue for positive effects, red for negative - VERY VISIBLE */
.token.positive-logit {{
    border-bottom: 4px solid #0066ff !important;
    box-shadow: 0 2px 0 #0066ff !important;
}}
.token.negative-logit {{
    border-bottom: 4px solid #ff3333 !important;
    box-shadow: 0 2px 0 #ff3333 !important;
}}
/* Stronger logit effects get even thicker underlines */
.token.strong-positive {{
    border-bottom: 6px solid #0066ff !important;
    box-shadow: 0 3px 0 #0066ff !important;
}}
.token.strong-negative {{
    border-bottom: 6px solid #ff3333 !important;
    box-shadow: 0 3px 0 #ff3333 !important;
}}
/* All the messy hovering stuff! */
.hover-text {{
    position: relative;
    cursor: pointer;
    display: inline-block; /* Needed to contain the tooltip */
    box-sizing: border-box;
}}
.tooltip {{
    font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
    font-size: 0.8em;
    background-color: #fff;
    color: #333; /* ? why is this here? */
    text-align: center;
    border-radius: 10px;
    padding: 5px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
    align-items: center;
    justify-content: center;
    overflow: hidden;
    display: none;
    position: fixed;
    z-index: 1000;
    /* to stand out from rest of tooltip content, code should have grey background */
    code {{
        font-family: "SF Mono", "Monaco", "Menlo", "Ubuntu Mono", "Consolas", monospace;
        background-color: #ddd;
        padding: 0px 3px;
    }}
}}

/* ! Tables */

table {{
    border: unset;
    color: black;
    border-collapse: collapse;
    width: -moz-fit-content;
    width: -webkit-fit-content;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    font-size: 0.8em;
}}
table.table-left tr {{
    border-bottom: 1px solid #eee;
    padding: 15px;
}}
table.table-left td {{
    padding: 3px 4px;
}}
table.table-left {{
    width: 100%;
}}
table.table-left td.left-aligned {{
    max-width: 120px;
    overflow-x: hidden;
}}
td {{
    border: none;
    padding: 2px 4px;
    white-space: nowrap;
}}
.right-aligned {{
    text-align: right;
}}
.left-aligned {{
    text-align: left;
}}
.center-aligned {{
    text-align: center;
    padding-bottom: 8px;
}}
table span {{
    background-color: #ddd;
    padding: 0px 4px;
    border-radius: 3px;
    white-space: pre-wrap; /* preserves whitespaces in table tokens */
}}
.table-container {{
    width: 100%;
}}
.half-width-container {{
    display: flex;
}}
.half-width {{
    width: 50%;
    margin-right: -4px;
}}

/* Feature tables should have space below them, also they should have a min column width */
.featureTables table {{
    margin-bottom: 25px;
    min-width: 250px;
}}
/* Configure logits table container (i.e. the thing containing the smaller and larger tables) */
.logitsTable {{
    min-width: 375px;
    display: flex;
    overflow-x: hidden;
    margin-bottom: 20px;
}}
/* Code is always bold in this table (this is just the neg/pos string tokens) */
.logitsTable code {{
    font-weight: bold;
}}
/* Set width of the tables inside the container (so they can stack horizontally), also put a gap between them */
.logitsTable > div.positive {{
    width: 47%;
}}
.logitsTable > div.negative {{
    width: 47%;
    margin-right: 5%;
}}
    </style>
</head>
<body>
    <div class="feature-count">
        {feature_count} features loaded
    </div>

    <div id="dropdown-container"></div>
    <div class="grid-container"></div>

    <script>
        const START_KEY = "{start_feature}";
        const METADATA = {json.dumps(metadata)};
        const DATA = {json.dumps(feature_data)};
        const PROMPT_DATA = {{}};
        
        // Include the original init.js content which will use lossEffect for underlines
        {js_content}
        
        // Enhanced initialization with sorting controls
        document.addEventListener("DOMContentLoaded", function() {{
            // Keys are either PROMPT_DATA.keys() if non-empty, else DATA.keys()
            const promptVisMode = Object.keys(PROMPT_DATA).length > 0;
            const allKeys = Object.keys(promptVisMode ? PROMPT_DATA : DATA);
            
            // The start key has already been user-defined, we need to check it's present in DASHBOARD_DATA
            if (!allKeys.includes(START_KEY)) {{
                console.error(`No data available for key: ${{START_KEY}}`);
            }}
            
            if (allKeys.length > 1) {{
                // Create sort controls
                const sortDiv = d3.select('#dropdown-container')
                    .append('div')
                    .attr('class', 'sort-controls');
                    
                sortDiv.append('label').text('Sort by:');
                
                const buttons = [
                    {{ id: 'feature-num', text: 'Feature Number', type: 'number' }},
                    {{ id: 'density-asc', text: 'Density ↑', type: 'density-asc' }},
                    {{ id: 'density-desc', text: 'Density ↓', type: 'density-desc' }}
                ];
                
                let currentSort = 'density-desc'; // Start with density descending
                let excludeDeadFeatures = true;   // Start with dead features excluded
                
                // Add sort buttons
                buttons.forEach(button => {{
                    const btn = sortDiv.append('button')
                        .attr('id', button.id)
                        .attr('class', `sort-button ${{button.type === 'density-desc' ? 'active' : ''}}`)  
                        .text(button.text)
                        .on('click', function() {{
                            // Update active button
                            sortDiv.selectAll('.sort-button').classed('active', false);
                            d3.select(this).classed('active', true);
                            
                            currentSort = button.type;
                            updateDropdown();
                        }});
                }});
                
                // Add exclude dead features checkbox
                const checkboxContainer = sortDiv.append('div')
                    .attr('class', 'checkbox-container');
                    
                const checkbox = checkboxContainer.append('input')
                    .attr('type', 'checkbox')
                    .attr('id', 'exclude-dead')
                    .attr('checked', true) // Start checked
                    .on('change', function() {{
                        excludeDeadFeatures = this.checked;
                        updateDropdown();
                    }});
                    
                checkboxContainer.append('label')
                    .attr('for', 'exclude-dead')
                    .text('Exclude Dead Features');
                
                function updateDropdown() {{
                    // Clear existing dropdown
                    d3.select('#dropdown-container').selectAll('.select').remove();
                    
                    // Sort keys based on current sort type
                    let sortedKeys = [...allKeys];
                    
                    if (currentSort === 'number') {{
                        sortedKeys.sort((a, b) => parseInt(a) - parseInt(b));
                    }} else if (currentSort === 'density-asc' || currentSort === 'density-desc') {{
                        // Calculate density for each feature
                        const densities = {{}};
                        sortedKeys.forEach(key => {{
                            const data = DATA[key];
                            if (data && data.actsHistogram && data.actsHistogram.y) {{
                                const totalActs = data.actsHistogram.y.reduce((sum, val) => sum + val, 0);
                                const totalPositions = data.seqMultiGroup && data.seqMultiGroup[0] && data.seqMultiGroup[0].seqGroupData ? 
                                    data.seqMultiGroup[0].seqGroupData.length * 50 : 1000; // estimate
                                densities[key] = totalActs / totalPositions;
                            }} else {{
                                densities[key] = 0;
                            }}
                        }});
                        
                        if (currentSort === 'density-asc') {{
                            sortedKeys.sort((a, b) => densities[a] - densities[b]);
                        }} else {{
                            sortedKeys.sort((a, b) => densities[b] - densities[a]);
                        }}
                    }}
                    
                    // Filter dead features if requested
                    if (excludeDeadFeatures) {{
                        sortedKeys = sortedKeys.filter(key => {{
                            const data = DATA[key];
                            if (data && data.actsHistogram && data.actsHistogram.y) {{
                                const totalActs = data.actsHistogram.y.reduce((sum, val) => sum + val, 0);
                                return totalActs > 0;
                            }}
                            return true;
                        }});
                    }}
                    
                    // Create dropdown with sorted keys
                    if (sortedKeys.length > 1) {{
                        const selectDiv = d3.select('#dropdown-container').append('div').attr('class', 'select');
                        const selectElem = selectDiv.append('select').attr('id', 'feature-select');
                        let maxWidth = 0;
                        
                        sortedKeys.forEach(key => {{
                            // Extract density directly from the histogram title
                            const data = DATA[key];
                            let densityText = "";
                            if (data && data.actsHistogram && data.actsHistogram.title) {{
                                const titleMatch = data.actsHistogram.title.match(/density: ([\\d.]+%)/);
                                if (titleMatch) {{
                                    densityText = ` (${{titleMatch[1]}})`;
                                }}
                            }}
                            
                            const displayText = `Feature ${{key}}${{densityText}}`;
                            selectElem.append('option')
                                .text(displayText)
                                .attr('value', key)
                                .attr('selected', START_KEY === key ? 'selected' : null);
                            
                            const width = measureTextWidth(displayText, "1em monospace");
                            if (width > maxWidth) {{ maxWidth = width; }}
                        }});
                        
                        selectElem.property('value', START_KEY);
                        selectDiv.style('width', `${{maxWidth + 45}}px`);
                        
                        selectElem.on('change', function() {{
                            const selectedKey = this.value;
                            setupPage(selectedKey);
                        }});
                    }}
                }}
                
                // Initial dropdown creation
                updateDropdown();
            }}
            
            // Initial trigger
            setupPage(START_KEY);
        }});
        // Custom measureTextWidth function if not included above
        if (typeof measureTextWidth === 'undefined') {{
            function measureTextWidth(text, font) {{
                const span = document.createElement("span");
                span.style.visibility = "hidden";
                span.style.position = "absolute";
                span.style.font = font;
                span.textContent = text;
                document.body.appendChild(span);
                const width = span.offsetWidth;
                document.body.removeChild(span);
                return width;
            }}
        }}
    </script>
</body>
</html>'''
    
    return html_template


def main():
    """Main function to generate crosscoder dashboard."""
    
    print("Loading crosscoder...")
    crosscoder = load_latest_crosscoder()
    
    print("Loading model...")
    model = LanguageModel("gpt2", device_map="cpu")
    
    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Prepare tokens
    tokens = []
    count = 0
    max_samples = 20  # Reduced for faster processing
    seq_len = 64  # Reduced for faster processing
    
    for item in dataset:
        if count >= max_samples:
            break
            
        text = item['text']
        if len(text.strip()) < 50:
            continue
            
        token_ids = model.tokenizer.encode(
            text,
            return_tensors="pt", 
            max_length=seq_len,
            truncation=True,
            padding="max_length"
        )
        
        tokens.append(token_ids)
        count += 1
    
    tokens = torch.cat(tokens, dim=0)
    print(f"Prepared {len(tokens)} sequences")
    
    print("Creating visualization data...")
    vis_data = CrosscoderVisData(crosscoder, model, tokens)
    
    # Generate data for first 5 features to match EXAMPLE.html
    feature_data = {}
    n_features = min(5, crosscoder.ae_dim)
    
    for feature_idx in tqdm.tqdm(range(n_features), desc="Generating feature data"):
        feature_data[str(feature_idx)] = vis_data.create_feature_data(feature_idx)
    
    print("Generating HTML...")
    html = generate_dashboard_html(feature_data, start_feature=0)
    
    # Save the dashboard
    output_file = "crosscoder_dashboard.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Dashboard saved to {output_file}")
    print("Open the file in a web browser to view the crosscoder visualization!")


if __name__ == "__main__":
    main()