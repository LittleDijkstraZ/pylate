#!/usr/bin/env python3
"""
Plot compression experiment results.

Usage:
    python plot_moderncolbert_results.py RESULTS_FILE [RESULTS_FILE2 ...]

Example:
    python plot_moderncolbert_results.py results/compression_experiments_new2/jinaai_jina-colbert-v2/amazon_dataset/beir_format/experiment_20251221_194453/results_20251221_194501.jsonl
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

def extract_compression_info(result):
    """Extract compression method and keep_ratio from result.

    Args:
        result: The result dict, which may contain a 'config' key for training-free
                compression or 'num_select_tokens' for ProxyAttentionColBERT.

    Returns:
        Tuple of (method_name, keep_ratio, pool_factor)
    """
    # Check if this is a ProxyAttentionColBERT result
    if 'num_select_tokens' in result and 'num_proxy_tokens' in result:
        num_select = result['num_select_tokens']
        num_proxy = result['num_proxy_tokens']
        return f'ProxyAttention (P={num_proxy})', None, None

    # Handle training-free compression results with 'config' key
    config = result.get('config', {})

    if config.get('type') == 'baseline':
        return 'Baseline', 1.0, None

    strategies = config.get('strategies', [])
    if not strategies:
        return 'Unknown', 1.0, None

    strategy = strategies[0]
    strategy_type = strategy['type']
    strategy_config = strategy.get('config', {})

    # Get keep_ratio or pool_factor
    keep_ratio = strategy_config.get('keep_ratio')
    pool_factor = strategy_config.get('pool_factor')

    # Map strategy types to readable names
    type_map = {
        'random_pruning': 'Random Pruning',
        'random_pooling': 'Random Pooling',
        'attention_pruning': 'Attention Pruning',
        'attention_pooling': 'Attention Pooling',
        'leverage-score': 'Leverage Score',
        'idf_pruning': 'IDF Pruning',
        'pooling': 'Clustering Pooling'
    }

    method_name = type_map.get(strategy_type, strategy_type)

    # For clustering pooling, add the method
    if strategy_type == 'pooling':
        clustering_method = strategy_config.get('clustering_method', 'unknown')
        method_name = f"{clustering_method.capitalize()} Pooling"

    return method_name, keep_ratio, pool_factor

def plot_results(datasets_results, output_dir='results/compression_new_plots', model_name=None):
    """Create comprehensive plots for the results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Metrics to plot
    metrics = ['ndcg@10', 'recall@10', 'map', 'mrr@10', 'precision@10']

    # Group results by compression method
    for dataset_name, (metadata, results) in datasets_results.items():
        print(f"\nProcessing {dataset_name}...")

        # Get model name from metadata if not provided
        display_model_name = model_name or metadata.get('model_name', 'Unknown Model')

        # Organize by method and keep_ratio
        method_data = defaultdict(lambda: {'keep_ratios': [], 'compression_ratios': [], 'avg_tokens': [], 'metrics': defaultdict(list)})

        baseline_tokens = None
        baseline_metrics = {}

        for result in results:
            method, keep_ratio, pool_factor = extract_compression_info(result)

            if method == 'Baseline':
                baseline_tokens = result['avg_tokens_per_doc']
                baseline_metrics = result['metrics']
                continue

            # Calculate compression ratio
            avg_tokens = result['avg_tokens_per_doc']
            compression_ratio = avg_tokens / baseline_tokens if baseline_tokens else 0

            method_data[method]['keep_ratios'].append(keep_ratio if keep_ratio else 1.0/pool_factor if pool_factor else 1.0)
            method_data[method]['compression_ratios'].append(compression_ratio)
            method_data[method]['avg_tokens'].append(avg_tokens)

            for metric in metrics:
                if metric in result['metrics']:
                    method_data[method]['metrics'][metric].append(result['metrics'][metric])

        # Create plots for each metric
        for metric in metrics:
            if not any(data['metrics'].get(metric) for data in method_data.values()):
                print(f"  Skipping {metric} - no data available")
                continue

            fig, ax = plt.subplots(figsize=(9, 7))

            # Plot each method
            for method, data in sorted(method_data.items()):
                if not data['avg_tokens'] or not data['metrics'].get(metric):
                    continue

                # Sort by avg_tokens
                sorted_indices = np.argsort(data['avg_tokens'])
                avg_tokens_sorted = np.array(data['avg_tokens'])[sorted_indices]
                metric_values = np.array(data['metrics'][metric])[sorted_indices]

                ax.plot(avg_tokens_sorted, metric_values, marker='o', label=method, linewidth=2, markersize=6)

            # Add baseline
            if baseline_metrics and metric in baseline_metrics:
                ax.axhline(y=baseline_metrics[metric], color='black', linestyle='--', linewidth=2, label='Baseline (No Compression)')

            # Create secondary x-axis for compression ratio
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            # Set secondary axis ticks to show compression ratio
            if baseline_tokens:
                ax_min, ax_max = ax.get_xlim()
                comp_min = (ax_min / baseline_tokens) * 100
                comp_max = (ax_max / baseline_tokens) * 100
                ax2.set_xlim(comp_min, comp_max)
                ax2.set_xlabel('Compression Ratio (%)', fontsize=12)

            ax.set_xlabel('Average Tokens per Document', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{dataset_name}: {metric.upper()} vs Token Length\nModel: {display_model_name}', fontsize=14, pad=40)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            # Use sanitized names for output files
            safe_dataset = dataset_name.replace('/', '_').replace('\\', '_')
            safe_model = display_model_name.replace('/', '_').replace('\\', '_')
            output_path = Path(output_dir) / f'{safe_model}_{safe_dataset}_{metric.replace("@", "_at_")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()
            del fig

def plot_comparison(datasets_results, output_dir='results/compression_new_plots'):
    """Create side-by-side comparison plots for all three datasets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = ['ndcg@10', 'recall@10', 'map']
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, (dataset_name, (metadata, results)) in enumerate(sorted(datasets_results.items())):
            ax = axes[idx]
            
            # Organize by method
            method_data = defaultdict(lambda: {'avg_tokens': [], 'metrics': defaultdict(list)})
            baseline_tokens = None
            baseline_metrics = {}

            for result in results:
                method, keep_ratio, pool_factor = extract_compression_info(result)

                if method == 'Baseline':
                    baseline_tokens = result['avg_tokens_per_doc']
                    baseline_metrics = result['metrics']
                    continue

                avg_tokens = result['avg_tokens_per_doc']

                method_data[method]['avg_tokens'].append(avg_tokens)
                method_data[method]['metrics'][metric].append(result['metrics'][metric])

            # Plot each method
            for method, data in sorted(method_data.items()):
                if not data['avg_tokens']:
                    continue

                sorted_indices = np.argsort(data['avg_tokens'])
                avg_tokens_sorted = np.array(data['avg_tokens'])[sorted_indices]
                metric_values = np.array(data['metrics'][metric])[sorted_indices]

                ax.plot(avg_tokens_sorted, metric_values, marker='o', label=method, linewidth=2, markersize=6)

            # Add baseline
            if baseline_metrics:
                ax.axhline(y=baseline_metrics[metric], color='black', linestyle='--', linewidth=2, label='Baseline')

            ax.set_xlabel('Avg Tokens/Doc', fontsize=11)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{dataset_name.capitalize()}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{metric.upper()} vs Compression Ratio - lightonai/GTE-ModernColBERT-v1', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = Path(output_dir) / f'moderncolbert_comparison_{metric.replace("@", "_at_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Plot compression experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python plot_moderncolbert_results.py results/.../results_20251221_194501.jsonl
    python plot_moderncolbert_results.py results1.jsonl results2.jsonl --output_dir my_plots
        """
    )
    parser.add_argument(
        'results_files',
        nargs='*',
        help='Path(s) to results JSONL file(s). If not provided, uses default paths.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots. Default: same directory as first results file'
    )

    args = parser.parse_args()

    # If no results files provided, use defaults
    if not args.results_files:
        datasets = {
            'nfcorpus': 'results/compression_experiments_new/lightonai_GTE-ModernColBERT-v1/nfcorpus/experiment_20251208_020154/results_20251208_020217.jsonl',
            'arguana': 'results/compression_experiments_new/lightonai_GTE-ModernColBERT-v1/arguana/experiment_20251208_020154/results_20251208_020226.jsonl',
            'scifact': 'results/compression_experiments_new/lightonai_GTE-ModernColBERT-v1/scifact/experiment_20251208_020154/results_20251208_020217.jsonl'
        }
    else:
        # Build datasets dict from provided files
        # When multiple files have the same dataset name, we combine their results
        datasets = {}  # dataset_name -> list of file paths
        for path in args.results_files:
            path = Path(path)
            if not path.exists():
                print(f"Warning: Results file not found: {path}")
                continue
            # Use dataset_name from metadata as key
            metadata, _ = load_results(path)
            dataset_name = metadata.get('dataset_name', path.stem)
            # Clean up dataset name for display
            if '/' in dataset_name:
                dataset_name = Path(dataset_name).name
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            datasets[dataset_name].append(str(path))

    if not datasets:
        print("Error: No valid results files found")
        return

    datasets_results = {}
    for name, paths in datasets.items():
        try:
            combined_results = []
            metadata = None
            for path in paths:
                meta, results = load_results(path)
                if metadata is None:
                    metadata = meta
                combined_results.extend(results)
            datasets_results[name] = (metadata, combined_results)
            print(f"Loaded {name}: {len(combined_results)} configurations from {len(paths)} file(s)")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue

    if not datasets_results:
        print("Error: No results loaded successfully")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use the directory of the first results file
        first_path = Path(list(datasets.values())[0][0])  # Get first path from first list
        output_dir = first_path.parent / 'plots'

    print(f"\nOutput directory: {output_dir}")

    # Create individual plots
    plot_results(datasets_results, output_dir=output_dir)

    # Create comparison plots only if we have multiple datasets
    if len(datasets_results) >= 3:
        print("\nCreating comparison plots...")
        plot_comparison(datasets_results, output_dir=output_dir)

    print("\n✓ All plots generated successfully!")

if __name__ == '__main__':
    main()

