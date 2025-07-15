import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from crosscoder.crosscoder import Crosscoder, Buffer, cc_config
from nnsight import LanguageModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datasets import load_dataset


class CrosscoderDashboard:
    def __init__(self, model_path: str, config_path: str):
        """Load trained crosscoder and initialize dashboard"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Convert dtype string back to torch dtype
        if isinstance(self.config['dtype'], str):
            self.config['dtype'] = getattr(torch, self.config['dtype'].split('.')[-1])
        
        # Initialize crosscoder
        self.crosscoder = Crosscoder(self.config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.config['device'])
        self.crosscoder.W_enc.data = checkpoint['W_enc']
        self.crosscoder.W_dec.data = checkpoint['W_dec']
        self.crosscoder.b_enc.data = checkpoint['b_enc']
        self.crosscoder.b_dec.data = checkpoint['b_dec']
        
        # Initialize model for getting activations
        self.model = LanguageModel("gpt2", device_map="auto")
        self.buffer = Buffer(self.config)
        
        # Cache for computed activations
        self.activation_cache = {}
        
    def get_sample_activations(self, n_samples: int = 1000) -> torch.Tensor:
        """Get sample activations for analysis"""
        if 'sample_acts' not in self.activation_cache:
            # Get fresh batch from buffer
            self.activation_cache['sample_acts'] = self.buffer.next()[:n_samples]
        return self.activation_cache['sample_acts']
    
    def analyze_feature(self, feature_idx: int, n_samples: int = 1000) -> Dict:
        """Analyze a specific feature"""
        acts = self.get_sample_activations(n_samples)
        
        # Get feature activations
        feature_acts = self.crosscoder.encode(acts)[:, feature_idx]
        
        # Get top activating examples
        top_indices = torch.argsort(feature_acts, descending=True)
        
        # Calculate statistics
        stats = {
            'density': (feature_acts > 0).float().mean().item(),
            'max_activation': feature_acts.max().item(),
            'mean_activation': feature_acts.mean().item(),
            'top_activations': feature_acts[top_indices[:20]].tolist(),
            'top_indices': top_indices[:20].tolist()
        }
        
        return stats
    
    def get_decoder_analysis(self, feature_idx: int) -> Dict:
        """Analyze decoder weights for a feature"""
        decoder_weights = self.crosscoder.W_dec[feature_idx]  # shape: (n_layers, d_model)
        
        # Calculate norms per layer
        layer_norms = decoder_weights.norm(dim=-1)
        
        # Get strongest and weakest layers
        strongest_layers = torch.argsort(layer_norms, descending=True)
        
        return {
            'layer_norms': layer_norms.tolist(),
            'strongest_layers': strongest_layers[:5].tolist(),
            'weakest_layers': strongest_layers[-5:].tolist(),
            'total_norm': layer_norms.sum().item()
        }
    
    def get_text_examples(self, feature_idx: int, n_examples: int = 10) -> List[str]:
        """Get text examples that activate the feature"""
        # Get some text data
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        
        examples = []
        for i, item in enumerate(dataset):
            if i >= n_examples * 3:  # Get more than needed
                break
            
            text = item['text']
            if len(text.strip()) < 50:
                continue
                
            # Tokenize and get activations
            tokens = self.model.tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            if tokens.shape[1] < 10:
                continue
                
            with self.model.trace(tokens) as tracer:
                layer_outputs = []
                for layer_idx in range(self.crosscoder.num_layers):
                    layer_out = self.model.transformer.h[layer_idx].output[0].save()
                    layer_outputs.append(layer_out)
            
            # Stack and process activations
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
            
            # Find maximum activation position
            max_pos = torch.argmax(feature_acts)
            max_activation = feature_acts[max_pos].item()
            
            if max_activation > 0.1:  # Only keep examples with meaningful activation
                # Get the token that maximally activated
                token_text = self.model.tokenizer.decode(tokens[0][max_pos + 1])  # +1 for BOS
                
                examples.append({
                    'text': text[:500],  # Truncate for display
                    'max_activation': max_activation,
                    'activating_token': token_text,
                    'position': max_pos.item()
                })
            
            if len(examples) >= n_examples:
                break
                
        return sorted(examples, key=lambda x: x['max_activation'], reverse=True)
    
    def create_activation_histogram(self, feature_idx: int) -> go.Figure:
        """Create activation histogram for a feature"""
        acts = self.get_sample_activations()
        feature_acts = self.crosscoder.encode(acts)[:, feature_idx]
        
        # Filter to only positive activations for histogram
        positive_acts = feature_acts[feature_acts > 0].detach().cpu().numpy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=positive_acts,
            nbinsx=50,
            name=f'Feature {feature_idx}',
            marker_color='orange',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Activation Histogram - Feature {feature_idx}',
            xaxis_title='Activation Value',
            yaxis_title='Count',
            showlegend=False
        )
        
        return fig
    
    def create_layer_norm_plot(self, feature_idx: int) -> go.Figure:
        """Create layer norm visualization"""
        decoder_analysis = self.get_decoder_analysis(feature_idx)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(len(decoder_analysis['layer_norms']))),
            y=decoder_analysis['layer_norms'],
            marker_color='blue',
            opacity=0.7,
            name='Layer Norms'
        ))
        
        fig.update_layout(
            title=f'Decoder Layer Norms - Feature {feature_idx}',
            xaxis_title='Layer Index',
            yaxis_title='Norm Value',
            showlegend=False
        )
        
        return fig
    
    def generate_feature_report(self, feature_idx: int) -> str:
        """Generate a complete report for a feature"""
        stats = self.analyze_feature(feature_idx)
        decoder_analysis = self.get_decoder_analysis(feature_idx)
        examples = self.get_text_examples(feature_idx, n_examples=5)
        
        report = f"""
        # Feature {feature_idx} Analysis
        
        ## Statistics
        - Activation Density: {stats['density']:.3f}%
        - Maximum Activation: {stats['max_activation']:.3f}
        - Mean Activation: {stats['mean_activation']:.3f}
        
        ## Decoder Analysis
        - Total Decoder Norm: {decoder_analysis['total_norm']:.3f}
        - Strongest Layers: {decoder_analysis['strongest_layers']}
        - Weakest Layers: {decoder_analysis['weakest_layers']}
        
        ## Top Activating Examples
        """
        
        for i, example in enumerate(examples[:3]):
            report += f"""
        ### Example {i+1} (Activation: {example['max_activation']:.3f})
        Token: "{example['activating_token']}"
        Text: {example['text'][:200]}...
        """
        
        return report
    
    def create_dashboard(self, feature_idx: int) -> str:
        """Create full dashboard HTML for a feature"""
        stats = self.analyze_feature(feature_idx)
        decoder_analysis = self.get_decoder_analysis(feature_idx)
        
        # Create plots
        hist_fig = self.create_activation_histogram(feature_idx)
        layer_fig = self.create_layer_norm_plot(feature_idx)
        
        # Convert to HTML
        hist_html = hist_fig.to_html(full_html=False, include_plotlyjs='cdn')
        layer_html = layer_fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Get text examples
        examples = self.get_text_examples(feature_idx, n_examples=5)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crosscoder Feature {feature_idx} Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .stat-box {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .example {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
                .activation {{ color: #d2691e; font-weight: bold; }}
                .token {{ background: #ffeb3b; padding: 2px 4px; border-radius: 3px; }}
                h1, h2 {{ color: #333; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            </style>
        </head>
        <body>
            <h1>Crosscoder Feature {feature_idx} Dashboard</h1>
            
            <div class="grid">
                <div>
                    <h2>Activation Statistics</h2>
                    <div class="stat-box">
                        <strong>Density:</strong> {stats['density']:.1%}<br>
                        <strong>Max Activation:</strong> {stats['max_activation']:.3f}<br>
                        <strong>Mean Activation:</strong> {stats['mean_activation']:.3f}
                    </div>
                </div>
                
                <div>
                    <h2>Decoder Analysis</h2>
                    <div class="stat-box">
                        <strong>Total Norm:</strong> {decoder_analysis['total_norm']:.3f}<br>
                        <strong>Strongest Layers:</strong> {decoder_analysis['strongest_layers']}<br>
                        <strong>Weakest Layers:</strong> {decoder_analysis['weakest_layers']}
                    </div>
                </div>
            </div>
            
            <div class="grid">
                <div>
                    <h2>Activation Distribution</h2>
                    {hist_html}
                </div>
                
                <div>
                    <h2>Layer Norms</h2>
                    {layer_html}
                </div>
            </div>
            
            <h2>Top Activating Examples</h2>
        """
        
        for i, example in enumerate(examples):
            html_content += f"""
            <div class="example">
                <h3>Example {i+1}</h3>
                <p><strong>Activation:</strong> <span class="activation">{example['max_activation']:.3f}</span></p>
                <p><strong>Token:</strong> <span class="token">{example['activating_token']}</span></p>
                <p><strong>Text:</strong> {example['text'][:300]}...</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content


def main():
    """Main function to create dashboard"""
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
    dashboard = CrosscoderDashboard(str(latest_checkpoint), str(config_file))
    
    # Generate dashboard for a few features
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    for feature_idx in range(5):  # First 5 features
        print(f"Generating dashboard for feature {feature_idx}...")
        html_content = dashboard.create_dashboard(feature_idx)
        
        output_file = output_dir / f"feature_{feature_idx}_dashboard.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to: {output_file}")
    
    print("Dashboard generation complete!")


if __name__ == "__main__":
    main()