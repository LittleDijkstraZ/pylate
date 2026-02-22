#!/usr/bin/env python3
"""
Plot Hierarchical Pooling + Attention Pooling comparison across different configurations.

Usage:
    python plot_pooling_comparison.py FILE1 FILE2 FILE3 --output_dir plots/

Example:
    python plot_pooling_comparison.py \
        results/.../beir_format/results.jsonl \
        results/.../beir_format_full/results1.jsonl \
        results/.../beir_format_full/results2.jsonl
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(jsonl_path):
    """Load results from JSONL file."""
    results = []
    metadata = None
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['type'] == 'metadata':
                metadata = data
            elif data['type'] == 'result':
                results.append(data)
    
    return metadata, results


def extract_compression_info(config):
    """Extract compression method and parameters from config."""
    if config.get('type') == 'baseline':
        return 'Baseline', 1.0, None
    
    strategies = config.get('strategies', [])
    if not strategies:
        return 'Unknown', 1.0, None
    
    strategy = strategies[0]
    strategy_type = strategy['type']
    strategy_config = strategy.get('config', {})
    
    keep_ratio = strategy_config.get('keep_ratio')
    pool_factor = strategy_config.get('pool_factor')
    
    type_map = {
        'attention_pooling': 'Attention Pooling',
        'pooling': 'Clustering Pooling',
        'idf_pruning': 'IDF Pruning',
    }
    
    method_name = type_map.get(strategy_type, strategy_type)
    
    if strategy_type == 'pooling':
        clustering_method = strategy_config.get('clustering_method', 'unknown')
        method_name = f"{clustering_method.capitalize()} Pooling"
    
    return method_name, keep_ratio, pool_factor


def filter_pooling_results(results):
    """Filter to only include Hierarchical and Attention pooling methods."""
    filtered = []
    for result in results:
        method, _, _ = extract_compression_info(result['config'])
        if method in ['Hierarchical Pooling', 'Attention Pooling', 'IDF Pruning', 'Baseline']:
            filtered.append(result)
    return filtered


def plot_comparison(all_configs, output_dir, metric='ndcg@10'):
    """Create comparison plot showing pooling methods across configurations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7*1.3, 4*1.3))
    
    colors = {'Hierarchical Pooling': 'tab:blue', 'Attention Pooling': 'tab:orange', 'IDF Pruning': 'tab:green'}
    markers = {'Hierarchical Pooling': 'o', 'Attention Pooling': 's', 'IDF Pruning': '^'}
    linestyles = ['-', '--', ':']  # Different line styles for different configs
    
    for config_idx, (config_name, (metadata, results)) in enumerate(all_configs.items()):
        method_data = defaultdict(lambda: {'avg_tokens': [], 'metric_values': []})
        baseline_metric = None
        
        for result in filter_pooling_results(results):
            method, keep_ratio, pool_factor = extract_compression_info(result['config'])
            
            if method == 'Baseline':
                baseline_metric = result['metrics'].get(metric)
                continue
            
            if method not in colors:
                continue
                
            method_data[method]['avg_tokens'].append(result['avg_tokens_per_doc'])
            method_data[method]['metric_values'].append(result['metrics'].get(metric, 0))
        
        # Plot each method for this config
        for method, data in method_data.items():
            if not data['avg_tokens']:
                continue
            
            sorted_idx = np.argsort(data['avg_tokens'])
            x = np.array(data['avg_tokens'])[sorted_idx]
            y = np.array(data['metric_values'])[sorted_idx]
            
            label = f"{method} ({config_name})"
            ax.plot(x, y, marker=markers[method], color=colors[method],
                   linestyle=linestyles[config_idx % len(linestyles)],
                   linewidth=2, markersize=8, label=label, alpha=0.8)
        
        # Add baseline for this config
        if baseline_metric:
            ax.axhline(y=baseline_metric, color='gray', linestyle='--', 
                      alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Average Tokens per Document', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} vs Token Length\nHierarchical & Attention Pooling Comparison', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'pooling_comparison_{metric.replace("@", "_at_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Hierarchical and Attention Pooling across configurations')
    parser.add_argument('results_files', nargs='+', help='Path(s) to results JSONL files')
    parser.add_argument('--output_dir', type=str, default='results/pooling_comparison_plots')
    parser.add_argument('--metrics', nargs='+', default=['ndcg@10', 'recall@10', 'map', 'precision@10'])
    parser.add_argument('--labels', nargs='+', help='Custom labels for each config (e.g., "128tok" "256tok" "1024tok")')
    
    args = parser.parse_args()
    
    all_configs = {}
    for i, path in enumerate(args.results_files):
        metadata, results = load_results(path)
        label = args.labels[i] if args.labels and i < len(args.labels) else f"Config {i+1}"
        all_configs[label] = (metadata, results)
        print(f"Loaded {label}: {len(results)} results, baseline tokens: {results[0]['avg_tokens_per_doc']:.0f}")
    
    for metric in args.metrics:
        plot_comparison(all_configs, args.output_dir, metric=metric)
    
    print(f"\n✓ All plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()

