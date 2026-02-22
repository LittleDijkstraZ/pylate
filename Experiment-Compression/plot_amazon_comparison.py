#!/usr/bin/env python3
"""
Plot comparison of compression methods on Amazon dataset.
Combines: Training-free compression, ProxyAttention, and ConstBERT results.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Professional color palette - grouped by method type
COLORS = {
    # Baseline
    'Baseline': '#1a1a2e',  # Dark navy

    # Learned compression methods (warm colors - reds/oranges)
    'ProxyAttention': '#e63946',  # Vibrant red
    'ConstBERT': '#f4a261',  # Sandy orange

    # Pooling methods (cool colors - blues/teals)
    'Hierarchical Pooling': '#1d3557',  # Dark blue
    'Spherical Pooling': '#457b9d',  # Steel blue
    'IDF Pooling': '#2a9d8f',  # Teal
    'Random pooling': '#a8dadc',  # Light cyan
    'Attention score pooling': '#48cae4',  # Sky blue

    # Pruning methods (greens/earth tones)
    'Random pruning': '#95a5a6',  # Gray
    'Attention score pruning': '#e9c46a',  # Gold
    'Leverage score pruning': '#8d99ae',  # Slate gray
    'Doc-wise IDF pruning': '#606c38',  # Olive green
}

def load_jsonl(path):
    """Load JSONL file and return metadata and results."""
    metadata = None
    results = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'metadata':
                metadata = data
            elif data.get('type') == 'result':
                results.append(data)
    return metadata, results

def get_method_name(config_name):
    """Extract method name from config name."""
    if 'Hierarchical' in config_name:
        return 'Hierarchical Pooling'
    elif 'Spherical' in config_name:
        return 'Spherical Pooling'
    elif 'IDF Pooling' in config_name:
        return 'IDF Pooling'
    elif 'Random pooling' in config_name:
        return 'Random pooling'
    elif 'Attention score pooling' in config_name:
        return 'Attention score pooling'
    elif 'Random pruning' in config_name:
        return 'Random pruning'
    elif 'Attention score pruning' in config_name:
        return 'Attention score pruning'
    elif 'Leverage score' in config_name:
        return 'Leverage score pruning'
    elif 'IDF pruning' in config_name:
        return 'Doc-wise IDF pruning'
    elif 'ProxyAttention' in config_name:
        return 'ProxyAttention'
    elif 'ConstBERT' in config_name:
        return 'ConstBERT'
    return config_name

def plot_comparison(results_dir, output_dir, metrics=['ndcg@10', 'map', 'recall@10']):
    """Create comparison plots."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = []
    baseline = None
    
    # Training-free results
    tf_dir = results_dir / 'training_free'
    if tf_dir.exists():
        for f in tf_dir.glob('results_*.jsonl'):
            _, results = load_jsonl(f)
            for r in results:
                if r['config_name'] == 'Baseline':
                    baseline = r
                else:
                    all_results.append(r)
    
    # ProxyAttention results
    pa_dir = results_dir / 'ProxyAttention-P32-ckpt15000'
    if pa_dir.exists():
        for f in sorted(pa_dir.glob('results_*.jsonl'))[-1:]:  # Latest only
            _, results = load_jsonl(f)
            all_results.extend(results)
    
    # ConstBERT results
    cb_dir = results_dir / 'ConstBERT-transpose-C32-ckpt15000'
    if cb_dir.exists():
        for f in cb_dir.glob('results_*.jsonl'):
            _, results = load_jsonl(f)
            for r in results:
                r['config_name'] = 'ConstBERT (32 tokens)'
            all_results.extend(results)
    
    # Group by method
    method_data = {}
    for r in all_results:
        method = get_method_name(r['config_name'])
        if method not in method_data:
            method_data[method] = {'tokens': [], 'metrics': {m: [] for m in metrics}}
        method_data[method]['tokens'].append(r['avg_tokens_per_doc'])
        for m in metrics:
            method_data[method]['metrics'][m].append(r['metrics'].get(m, 0))
    
    # Markers for different method types
    MARKERS = {
        'ProxyAttention': 'D',  # Diamond
        'ConstBERT': 'p',  # Pentagon
        'Hierarchical Pooling': 'o',  # Circle
        'Spherical Pooling': 's',  # Square
        'IDF Pooling': '^',  # Triangle up
        'Random pooling': 'v',  # Triangle down
        'Attention score pooling': '<',  # Triangle left
        'Random pruning': 'x',  # X
        'Attention score pruning': '+',  # Plus
        'Leverage score pruning': '1',  # Tri down
        'Doc-wise IDF pruning': '2',  # Tri up
    }

    # Create plots
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot baseline
        if baseline:
            bl_tokens = baseline['avg_tokens_per_doc']
            bl_value = baseline['metrics'].get(metric, 0)
            ax.axhline(y=bl_value, color=COLORS['Baseline'], linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
            ax.scatter([bl_tokens], [bl_value], color=COLORS['Baseline'], s=200, marker='*', zorder=5,
                      edgecolors='white', linewidths=1.5, label=f'Baseline ({bl_tokens:.0f} tok)')

        # Plot each method
        for method, data in sorted(method_data.items()):
            if not data['tokens']:
                continue
            sorted_idx = np.argsort(data['tokens'])
            tokens = np.array(data['tokens'])[sorted_idx]
            values = np.array(data['metrics'][metric])[sorted_idx]
            color = COLORS.get(method, '#333333')
            marker = MARKERS.get(method, 'o')

            # Thicker lines for learned methods
            linewidth = 3 if method in ['ProxyAttention', 'ConstBERT'] else 2
            markersize = 12 if method in ['ProxyAttention', 'ConstBERT'] else 9

            ax.plot(tokens, values, marker=marker, label=method, linewidth=linewidth,
                   markersize=markersize, color=color, markeredgecolor='white',
                   markeredgewidth=0.5, alpha=0.9)

        ax.set_xlabel('Average Tokens per Document', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=13, fontweight='bold')
        ax.set_title(f'Amazon Dataset: {metric.upper()} vs Compression\nComparing Training-free, ProxyAttention, and ConstBERT',
                    fontsize=15, fontweight='bold', pad=15)

        # Better legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                 framealpha=0.95, edgecolor='gray', fancybox=True)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')

        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)

        plt.tight_layout()
        out_path = output_dir / f'amazon_comparison_{metric.replace("@", "_at_")}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot Amazon compression comparison')
    parser.add_argument('--results_dir', default='results/amazon_compression_comparison_20260209')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--metrics', nargs='+', default=['ndcg@10', 'map', 'recall@10', 'mrr@10'])
    args = parser.parse_args()
    
    output_dir = args.output_dir or f"{args.results_dir}/plots"
    plot_comparison(args.results_dir, output_dir, args.metrics)
    print(f"\n✓ All plots saved to {output_dir}")

if __name__ == '__main__':
    main()

