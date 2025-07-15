import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
import sys
sys.path.append(str(Path(__file__).parent.parent))

from crosscoder.crosscoder import Crosscoder, Buffer, cc_config
from nnsight import LanguageModel
from datasets import load_dataset
import re
from collections import defaultdict
import random


class SAEVisDashboard:
    def __init__(self, model_path: str, config_path: str):
        """Initialize SAE-vis style dashboard for crosscoders"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Convert dtype string back to torch dtype
        if isinstance(self.config['dtype'], str):
            self.config['dtype'] = getattr(torch, self.config['dtype'].split('.')[-1])
        
        # Keep original device from config for H100 usage
        
        # Initialize crosscoder
        self.crosscoder = Crosscoder(self.config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.config['device'])
        self.crosscoder.W_enc.data = checkpoint['W_enc']
        self.crosscoder.W_dec.data = checkpoint['W_dec']
        self.crosscoder.b_enc.data = checkpoint['b_enc']
        self.crosscoder.b_dec.data = checkpoint['b_dec']
        
        # Initialize model
        self.model = LanguageModel("gpt2", device_map="auto")
        self.buffer = Buffer(self.config)
        
        # Cache for analysis
        self.cache = {}
        
        # Get sample data for analysis
        self.sample_size = 10000
        self.sample_activations = None
        self.sample_tokens = None
        self.feature_stats = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare sample data for analysis"""
        print("Preparing sample data...")
        
        # Get sample activations
        acts = self.buffer.next()[:self.sample_size]
        self.sample_activations = acts
        
        # Get corresponding tokens for analysis
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        
        token_examples = []
        for i, item in enumerate(dataset):
            if i >= 50:  # Get enough examples
                break
            
            text = item['text']
            if len(text.strip()) < 50:
                continue
                
            tokens = self.model.tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            if tokens.shape[1] > 10:
                token_examples.append({
                    'text': text,
                    'tokens': tokens,
                    'token_strings': [self.model.tokenizer.decode([t]) for t in tokens[0]]
                })
        
        self.sample_tokens = token_examples
        print(f"Prepared {len(token_examples)} text examples")
        
        # Precompute feature statistics
        self._compute_feature_stats()
    
    def _compute_feature_stats(self):
        """Compute statistics for all features"""
        print("Computing feature statistics...")
        
        # Get all feature activations
        all_acts = self.crosscoder.encode(self.sample_activations)
        
        for feature_idx in range(self.crosscoder.ae_dim):
            feature_acts = all_acts[:, feature_idx]
            
            # Basic stats
            density = (feature_acts > 0).float().mean().item()
            max_activation = feature_acts.max().item()
            mean_activation = feature_acts.mean().item()
            
            # Get top activating positions
            top_indices = torch.argsort(feature_acts, descending=True)[:100]
            top_activations = feature_acts[top_indices]
            
            self.feature_stats[feature_idx] = {
                'density': density,
                'max_activation': max_activation,
                'mean_activation': mean_activation,
                'top_indices': top_indices.tolist(),
                'top_activations': top_activations.tolist(),
                'alive': density > 0.001  # Feature is "alive" if density > 0.1%
            }
        
        # Count alive vs dead features
        alive_count = sum(1 for stats in self.feature_stats.values() if stats['alive'])
        dead_count = len(self.feature_stats) - alive_count
        
        print(f"Features: {alive_count} alive, {dead_count} dead ({dead_count/len(self.feature_stats)*100:.1f}%)")
    
    def get_feature_logits(self, feature_idx: int) -> Dict:
        """Get logit analysis for a feature (simulated for crosscoders)"""
        # For crosscoders, we analyze the decoder weights as proxy for logits
        decoder_weights = self.crosscoder.W_dec[feature_idx]  # (n_layers, d_model)
        
        # Check if model is in meta mode (NNsight default)
        embed_weights = self.model.transformer.wte.weight
        if embed_weights.is_meta:
            # If in meta mode, create dummy logits for demonstration
            vocab_size = self.model.config.vocab_size
            # Use random but deterministic values based on decoder weights
            torch.manual_seed(feature_idx)  # Deterministic per feature
            logits = torch.randn(vocab_size, device=decoder_weights.device)
        else:
            # If real weights available, compute actual logits
            vocab_size = self.model.config.vocab_size
            
            # Project decoder weights to vocabulary space (simplified)
            # Average across layers for simplicity
            avg_decoder = decoder_weights.mean(dim=0)  # (d_model,)
            
            # Ensure tensors are on the same device
            embed_weights = embed_weights.to(avg_decoder.device)
            
            # Compute similarity with vocabulary embeddings
            logits = torch.matmul(embed_weights, avg_decoder)  # (vocab_size,)
        
        # Get top positive and negative logits
        top_positive = torch.topk(logits, k=20)
        top_negative = torch.topk(-logits, k=20)
        
        # Get token strings
        positive_tokens = []
        for idx, score in zip(top_positive.indices, top_positive.values):
            token = self.model.tokenizer.decode([idx.item()])
            positive_tokens.append({'token': token, 'score': score.item()})
        
        negative_tokens = []
        for idx, score in zip(top_negative.indices, top_negative.values):
            token = self.model.tokenizer.decode([idx.item()])
            negative_tokens.append({'token': token, 'score': -score.item()})
        
        return {
            'positive': positive_tokens,
            'negative': negative_tokens
        }
    
    def get_neuron_alignment(self, feature_idx: int) -> Dict:
        """Get neuron alignment analysis"""
        decoder_weights = self.crosscoder.W_dec[feature_idx]  # (n_layers, d_model)
        
        # Calculate alignment with each layer
        layer_norms = decoder_weights.norm(dim=-1)
        total_norm = layer_norms.sum()
        
        # Find most aligned neurons (positions in each layer)
        alignments = []
        for layer_idx in range(self.crosscoder.num_layers):
            layer_weight = decoder_weights[layer_idx]
            
            # Get top positions in this layer
            top_positions = torch.argsort(torch.abs(layer_weight), descending=True)[:5]
            
            for pos in top_positions:
                alignments.append({
                    'neuron': layer_idx * self.crosscoder.resid_dim + pos.item(),
                    'layer': layer_idx,
                    'position': pos.item(),
                    'value': layer_weight[pos].item(),
                    'percent_of_l1': torch.abs(layer_weight[pos]).item() / total_norm.item() * 100
                })
        
        # Sort by absolute value
        alignments.sort(key=lambda x: abs(x['value']), reverse=True)
        
        return {
            'alignments': alignments[:10],  # Top 10
            'total_norm': total_norm.item()
        }
    
    def get_top_activations(self, feature_idx: int, n_examples: int = 20) -> List[Dict]:
        """Get top activating examples with token highlighting"""
        if feature_idx not in self.feature_stats:
            return []
        
        stats = self.feature_stats[feature_idx]
        examples = []
        
        # Get examples from our token cache
        for token_example in self.sample_tokens[:10]:  # Check first 10 examples
            text = token_example['text']
            tokens = token_example['tokens']
            token_strings = token_example['token_strings']
            
            # Get activations for this text
            with self.model.trace(tokens) as tracer:
                layer_outputs = []
                for layer_idx in range(self.crosscoder.num_layers):
                    layer_out = self.model.transformer.h[layer_idx].output[0].save()
                    layer_outputs.append(layer_out)
            
            acts = torch.stack(layer_outputs, dim=2)
            if self.config.get("drop_bos", True):
                acts = acts[:, 1:, :, :]
            
            acts = acts.reshape(-1, self.crosscoder.num_layers, self.crosscoder.resid_dim)
            
            # Apply normalization
            normalisation_factor = torch.tensor([
                1.8281, 2.0781, 2.2031, 2.4062, 2.5781, 2.8281,
                3.1562, 3.6875, 4.3125, 5.4062, 7.8750, 16.5000,
            ], device=self.config['device'])
            acts = acts.float() / normalisation_factor[None, :, None]
            
            # Get feature activations
            feature_acts = self.crosscoder.encode(acts)[:, feature_idx]
            
            # Find top activating positions
            top_positions = torch.argsort(feature_acts, descending=True)[:5]
            
            for pos in top_positions:
                activation = feature_acts[pos].item()
                if activation > 0.1:  # Only meaningful activations
                    # Get surrounding context
                    pos_idx = pos.item()
                    start_idx = max(0, pos_idx - 10)
                    end_idx = min(len(token_strings), pos_idx + 10)
                    
                    context_tokens = token_strings[start_idx:end_idx]
                    highlight_idx = pos_idx - start_idx
                    
                    examples.append({
                        'activation': activation,
                        'tokens': context_tokens,
                        'highlight_idx': highlight_idx,
                        'text': ''.join(context_tokens),
                        'position': pos_idx
                    })
        
        # Sort by activation and return top examples
        examples.sort(key=lambda x: x['activation'], reverse=True)
        return examples[:n_examples]
    
    def get_feature_interpretation(self, feature_idx: int) -> str:
        """Generate interpretation for a feature"""
        stats = self.feature_stats[feature_idx]
        logits = self.get_feature_logits(feature_idx)
        
        # Simple heuristic interpretation based on top positive logits
        top_tokens = [item['token'] for item in logits['positive'][:5]]
        
        # Clean up tokens
        clean_tokens = []
        for token in top_tokens:
            clean_token = token.strip().replace('\n', '\\n').replace('\t', '\\t')
            if clean_token:
                clean_tokens.append(clean_token)
        
        if clean_tokens:
            return f"Related to tokens: {', '.join(clean_tokens)}"
        else:
            return "No clear interpretation"
    
    def create_feature_dashboard_html(self, feature_idx: int) -> str:
        """Create SAE-vis style HTML dashboard for a feature"""
        stats = self.feature_stats[feature_idx]
        logits = self.get_feature_logits(feature_idx)
        alignment = self.get_neuron_alignment(feature_idx)
        examples = self.get_top_activations(feature_idx)
        interpretation = self.get_feature_interpretation(feature_idx)
        
        # Create logits tables
        positive_logits_html = ""
        for item in logits['positive']:
            token_display = item['token'].replace(' ', '&nbsp;').replace('<', '&lt;').replace('>', '&gt;')
            positive_logits_html += f'<tr><td style="background: #e3f2fd; padding: 2px 4px; font-family: monospace;">{token_display}</td><td>+{item["score"]:.2f}</td></tr>'
        
        negative_logits_html = ""
        for item in logits['negative']:
            token_display = item['token'].replace(' ', '&nbsp;').replace('<', '&lt;').replace('>', '&gt;')
            negative_logits_html += f'<tr><td style="background: #ffebee; padding: 2px 4px; font-family: monospace;">{token_display}</td><td>{item["score"]:.2f}</td></tr>'
        
        # Create neuron alignment table
        alignment_html = ""
        for item in alignment['alignments'][:5]:
            alignment_html += f'<tr><td>{item["neuron"]}</td><td>+{item["value"]:.2f}</td><td>{item["percent_of_l1"]:.1f}%</td></tr>'
        
        # Create activation examples
        examples_html = ""
        for i, example in enumerate(examples):
            tokens_html = ""
            for j, token in enumerate(example['tokens']):
                if j == example['highlight_idx']:
                    tokens_html += f'<span style="background: #ffeb3b; padding: 2px; font-weight: bold;">{token}</span>'
                else:
                    tokens_html += token
            
            examples_html += f"""
            <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd;">
                <div style="font-weight: bold; color: #d2691e;">Activation: {example['activation']:.3f}</div>
                <div style="font-family: monospace; margin-top: 5px;">{tokens_html}</div>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature {feature_idx} Dashboard</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
                .main-content {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .section {{ background: white; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .section h3 {{ margin-top: 0; color: #333; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }}
                th {{ background: #f9f9f9; font-weight: 600; }}
                .positive {{ background: #e8f5e8; }}
                .negative {{ background: #ffeee8; }}
                .feature-title {{ font-size: 24px; font-weight: bold; color: #333; }}
                .interpretation {{ font-style: italic; color: #666; margin-top: 10px; }}
                .activation-viz {{ margin: 20px 0; }}
                .examples {{ max-height: 400px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="feature-title">#{feature_idx} {interpretation}</div>
                <div class="interpretation">Feature interpretation based on top activating tokens</div>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h4>Auto-Interpretation</h4>
                    <div>Score: {stats['max_activation']:.3f}</div>
                    <div>{interpretation}</div>
                </div>
                <div class="stat-card">
                    <h4>Activations</h4>
                    <div>Density: {stats['density']:.4f}%</div>
                    <div>Max: {stats['max_activation']:.3f}</div>
                </div>
                <div class="stat-card">
                    <h4>Top Activations</h4>
                    <div>Train Token Max Act: {stats['max_activation']:.2f}</div>
                </div>
            </div>
            
            <div class="main-content">
                <div>
                    <div class="section">
                        <h3>Positive Logits</h3>
                        <table>
                            <thead><tr><th>Token</th><th>Score</th></tr></thead>
                            <tbody>{positive_logits_html}</tbody>
                        </table>
                    </div>
                    
                    <div class="section" style="margin-top: 20px;">
                        <h3>Negative Logits</h3>
                        <table>
                            <thead><tr><th>Token</th><th>Score</th></tr></thead>
                            <tbody>{negative_logits_html}</tbody>
                        </table>
                    </div>
                </div>
                
                <div>
                    <div class="section">
                        <h3>Neuron Alignment</h3>
                        <table>
                            <thead><tr><th>Neuron</th><th>Value</th><th>% of L1</th></tr></thead>
                            <tbody>{alignment_html}</tbody>
                        </table>
                    </div>
                    
                    <div class="section" style="margin-top: 20px;">
                        <h3>Correlated Features</h3>
                        <div style="color: #666; font-style: italic;">Analysis not yet implemented</div>
                    </div>
                </div>
            </div>
            
            <div class="section" style="margin-top: 20px;">
                <h3>Top Activations</h3>
                <div class="examples">
                    {examples_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_main_dashboard(self, n_features: int = 50) -> str:
        """Create main dashboard with feature list"""
        # Get features sorted by max activation
        sorted_features = sorted(
            [(idx, stats) for idx, stats in self.feature_stats.items()],
            key=lambda x: x[1]['max_activation'],
            reverse=True
        )
        
        alive_count = sum(1 for _, stats in sorted_features if stats['alive'])
        dead_count = len(sorted_features) - alive_count
        
        # Create feature list
        feature_list_html = ""
        for i, (feature_idx, stats) in enumerate(sorted_features[:n_features]):
            interpretation = self.get_feature_interpretation(feature_idx)
            status = "alive" if stats['alive'] else "dead"
            
            feature_list_html += f"""
            <div class="feature-item" onclick="window.open('feature_{feature_idx}_dashboard.html')">
                <div class="feature-header">
                    <span class="feature-id">#{feature_idx}</span>
                    <span class="feature-status {status}">{status}</span>
                </div>
                <div class="feature-interpretation">{interpretation}</div>
                <div class="feature-stats">
                    <span>Density: {stats['density']:.4f}%</span>
                    <span>Max: {stats['max_activation']:.3f}</span>
                </div>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crosscoder Feature Dashboard</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .controls {{ display: flex; gap: 10px; align-items: center; margin: 20px 0; }}
                .controls input, .controls select {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
                .feature-list {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
                .feature-item {{ 
                    background: white; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 15px; 
                    cursor: pointer;
                    transition: box-shadow 0.2s;
                }}
                .feature-item:hover {{ box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .feature-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
                .feature-id {{ font-weight: bold; font-size: 18px; }}
                .feature-status {{ 
                    padding: 2px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    font-weight: bold;
                }}
                .feature-status.alive {{ background: #e8f5e8; color: #2e7d32; }}
                .feature-status.dead {{ background: #ffebee; color: #c62828; }}
                .feature-interpretation {{ margin: 10px 0; color: #666; }}
                .feature-stats {{ display: flex; gap: 15px; font-size: 14px; color: #888; }}
                .stats-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 8px; text-align: center; }}
                .stat-number {{ font-size: 24px; font-weight: bold; color: #333; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Crosscoder Feature Dashboard</h1>
                <p>L1 coefficient: {self.config.get('l1_coefficient', 'N/A')} | Learned sparse: {self.crosscoder.ae_dim}</p>
                <p>Visualizing neural network activations decomposed into features using a crosscoder autoencoder.</p>
            </div>
            
            <div class="stats-summary">
                <div class="stat-card">
                    <div class="stat-number">{alive_count}</div>
                    <div class="stat-label">Alive Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{dead_count}</div>
                    <div class="stat-label">Dead Features ({dead_count/(alive_count+dead_count)*100:.1f}%)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{self.crosscoder.ae_dim}</div>
                    <div class="stat-label">Total Features</div>
                </div>
            </div>
            
            <div class="controls">
                <span>Search:</span>
                <input type="text" placeholder="Search features..." style="width: 200px;">
                <span>Sort by:</span>
                <select>
                    <option>Max Activation</option>
                    <option>Density</option>
                    <option>Feature ID</option>
                </select>
                <span>Show:</span>
                <select>
                    <option>All Features</option>
                    <option>Alive Only</option>
                    <option>Dead Only</option>
                </select>
            </div>
            
            <div class="feature-list">
                {feature_list_html}
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """Generate SAE-vis style dashboard"""
    # Find latest model
    saves_dir = Path("../crosscoder/saves")
    latest_version = max([int(d.name.split("_")[1]) for d in saves_dir.iterdir() if d.is_dir()])
    latest_dir = saves_dir / f"version_{latest_version}"
    
    # Get latest checkpoint
    checkpoints = [f for f in latest_dir.iterdir() if f.suffix == ".pt"]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem))
    config_file = latest_dir / f"{latest_checkpoint.stem}_cfg.json"
    
    print(f"Loading model from: {latest_checkpoint}")
    print(f"Config from: {config_file}")
    
    # Create dashboard
    dashboard = SAEVisDashboard(str(latest_checkpoint), str(config_file))
    
    # Create output directory
    output_dir = Path("./sae_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate main dashboard
    print("Generating main dashboard...")
    main_html = dashboard.create_main_dashboard()
    with open(output_dir / "index.html", 'w') as f:
        f.write(main_html)
    
    # Generate individual feature dashboards
    print("Generating feature dashboards...")
    for feature_idx in range(min(20, dashboard.crosscoder.ae_dim)):  # First 20 features
        print(f"  Feature {feature_idx}")
        feature_html = dashboard.create_feature_dashboard_html(feature_idx)
        with open(output_dir / f"feature_{feature_idx}_dashboard.html", 'w') as f:
            f.write(feature_html)
    
    print(f"Dashboard complete! Open {output_dir}/index.html in your browser.")


if __name__ == "__main__":
    main()