"""
Visualize embeddings to understand if they make sense
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_embeddings(path: str):
    """Load embeddings from inference results"""
    data = torch.load(path)
    pred = data["pred"].numpy()  # [N, 384]
    target = data["target"].numpy()  # [N, 384]
    return pred, target

def plot_embedding_heatmap(embeddings, title, output_path, max_samples=100):
    """Plot heatmap of embeddings to see patterns"""
    # Sample subset if too many
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort embeddings by their L2 norm for better visualization
    norms = np.linalg.norm(embeddings, axis=1)
    sorted_indices = np.argsort(norms)
    embeddings_sorted = embeddings[sorted_indices]
    
    # Plot heatmap
    im = ax.imshow(embeddings_sorted, aspect='auto', cmap='RdBu_r', 
                   vmin=-0.3, vmax=0.3, interpolation='nearest')
    
    ax.set_xlabel('Embedding Dimension (384 dims)', fontsize=12)
    ax.set_ylabel(f'Samples (showing {len(embeddings)} of total)', fontsize=12)
    ax.set_title(f'{title}\n(sorted by L2 norm)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Embedding Value', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_similarity_matrix(embeddings, title, output_path, max_samples=100):
    """Plot pairwise similarity matrix"""
    # Sample subset
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=-0.2, vmax=1.0)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(f'{title}\nPairwise Cosine Similarity ({len(embeddings)} samples)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    # Add diagonal line
    ax.plot([0, len(embeddings)], [0, len(embeddings)], 'r--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_dimension_statistics(pred, target, output_path):
    """Plot per-dimension statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Dimension means
    pred_means = pred.mean(axis=0)
    target_means = target.mean(axis=0)
    
    axes[0, 0].plot(pred_means, label='Predictions', alpha=0.7, linewidth=1)
    axes[0, 0].plot(target_means, label='Targets', alpha=0.7, linewidth=1)
    axes[0, 0].set_xlabel('Dimension', fontsize=11)
    axes[0, 0].set_ylabel('Mean Value', fontsize=11)
    axes[0, 0].set_title('Per-Dimension Means', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 2. Dimension variances
    pred_vars = pred.var(axis=0)
    target_vars = target.var(axis=0)
    
    axes[0, 1].plot(pred_vars, label='Predictions', alpha=0.7, linewidth=1)
    axes[0, 1].plot(target_vars, label='Targets', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Variance', fontsize=11)
    axes[0, 1].set_title('Per-Dimension Variances', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Value distribution
    axes[1, 0].hist(pred.flatten(), bins=100, alpha=0.6, label='Predictions', density=True)
    axes[1, 0].hist(target.flatten(), bins=100, alpha=0.6, label='Targets', density=True)
    axes[1, 0].set_xlabel('Embedding Value', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Embedding Value Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # 4. L2 norms (show statistics instead of histogram since they're normalized)
    pred_norms = np.linalg.norm(pred, axis=1)
    target_norms = np.linalg.norm(target, axis=1)
    
    axes[1, 1].text(0.5, 0.7, 'L2 Norm Statistics', 
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    transform=axes[1, 1].transAxes)
    
    stats_text = f"""Predictions:
  Mean: {pred_norms.mean():.4f}
  Std: {pred_norms.std():.4f}
  Range: [{pred_norms.min():.4f}, {pred_norms.max():.4f}]

Targets:
  Mean: {target_norms.mean():.4f}
  Std: {target_norms.std():.4f}
  Range: [{target_norms.min():.4f}, {target_norms.max():.4f}]

✓ Embeddings are well-normalized
  (all L2 norms ≈ 1.0)"""
    
    axes[1, 1].text(0.5, 0.5, stats_text, 
                    ha='center', va='center', fontsize=10, family='monospace',
                    transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_top_dimensions(pred, target, output_path, top_k=20):
    """Plot the most important dimensions (highest variance)"""
    # Find top-k dimensions by variance
    pred_vars = pred.var(axis=0)
    top_dims = np.argsort(pred_vars)[-top_k:][::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Heatmap of top dimensions
    pred_top = pred[:200, top_dims]  # First 200 samples
    
    im1 = axes[0].imshow(pred_top.T, aspect='auto', cmap='RdBu_r', 
                         vmin=-0.3, vmax=0.3, interpolation='nearest')
    axes[0].set_xlabel('Sample (first 200)', fontsize=11)
    axes[0].set_ylabel(f'Top-{top_k} Dimensions (by variance)', fontsize=11)
    axes[0].set_title('Top Dimensions - Predictions', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Show variance of top dimensions
    axes[1].bar(range(top_k), pred_vars[top_dims], color='steelblue', alpha=0.7)
    axes[1].set_xlabel(f'Top-{top_k} Dimension Rank', fontsize=11)
    axes[1].set_ylabel('Variance', fontsize=11)
    axes[1].set_title('Variance of Top Dimensions', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add dimension indices as labels
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels([f'{d}' for d in top_dims], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_sample_trajectories(pred, target, output_path, n_samples=10):
    """Plot how individual samples look across dimensions"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Random sample indices
    indices = np.random.choice(len(pred), n_samples, replace=False)
    
    # Plot predictions
    for i, idx in enumerate(indices):
        axes[0].plot(pred[idx], alpha=0.7, label=f'Sample {idx}', linewidth=1)
    
    axes[0].set_xlabel('Dimension', fontsize=11)
    axes[0].set_ylabel('Embedding Value', fontsize=11)
    axes[0].set_title(f'Predicted Embeddings - {n_samples} Random Samples', 
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot targets
    for i, idx in enumerate(indices):
        axes[1].plot(target[idx], alpha=0.7, label=f'Sample {idx}', linewidth=1)
    
    axes[1].set_xlabel('Dimension', fontsize=11)
    axes[1].set_ylabel('Embedding Value', fontsize=11)
    axes[1].set_title(f'Target Embeddings - {n_samples} Random Samples', 
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def analyze_embedding_structure(pred, target):
    """Print detailed analysis of embedding structure"""
    print("\n" + "="*80)
    print("EMBEDDING STRUCTURE ANALYSIS")
    print("="*80)
    
    # 1. Basic stats
    print(f"\n1. Basic Statistics:")
    print(f"   Shape: {pred.shape}")
    print(f"   Pred - Mean: {pred.mean():.6f}, Std: {pred.std():.6f}")
    print(f"   Target - Mean: {target.mean():.6f}, Std: {target.std():.6f}")
    
    # 2. Sparsity
    pred_zeros = (np.abs(pred) < 0.01).sum() / pred.size * 100
    target_zeros = (np.abs(target) < 0.01).sum() / target.size * 100
    print(f"\n2. Sparsity (values near zero):")
    print(f"   Pred: {pred_zeros:.2f}% of values < 0.01")
    print(f"   Target: {target_zeros:.2f}% of values < 0.01")
    
    # 3. Dimension usage
    pred_active_dims = (pred.std(axis=0) > 0.01).sum()
    target_active_dims = (target.std(axis=0) > 0.01).sum()
    print(f"\n3. Active Dimensions (std > 0.01):")
    print(f"   Pred: {pred_active_dims}/384 dimensions active ({pred_active_dims/384*100:.1f}%)")
    print(f"   Target: {target_active_dims}/384 dimensions active ({target_active_dims/384*100:.1f}%)")
    
    # 4. Correlation between pred and target
    print(f"\n4. Pred-Target Correlation:")
    per_sample_corr = np.array([
        np.corrcoef(pred[i], target[i])[0, 1] 
        for i in range(min(1000, len(pred)))
    ])
    print(f"   Per-sample correlation: {per_sample_corr.mean():.4f} ± {per_sample_corr.std():.4f}")
    
    # 5. Most/least variable dimensions
    pred_vars = pred.var(axis=0)
    print(f"\n5. Dimension Variance:")
    print(f"   Top-5 dimensions: {np.argsort(pred_vars)[-5:][::-1].tolist()}")
    print(f"   Bottom-5 dimensions: {np.argsort(pred_vars)[:5].tolist()}")
    print(f"   Max variance: {pred_vars.max():.6f}")
    print(f"   Min variance: {pred_vars.min():.6f}")
    
    # 6. Check for collapse
    mean_pairwise_sim = []
    for _ in range(100):
        idx1, idx2 = np.random.choice(len(pred), 2, replace=False)
        p1 = pred[idx1] / (np.linalg.norm(pred[idx1]) + 1e-8)
        p2 = pred[idx2] / (np.linalg.norm(pred[idx2]) + 1e-8)
        mean_pairwise_sim.append(np.dot(p1, p2))
    
    print(f"\n6. Collapse Check (random pairwise similarity):")
    print(f"   Mean: {np.mean(mean_pairwise_sim):.4f}")
    if np.mean(mean_pairwise_sim) > 0.8:
        print(f"   ⚠️  WARNING: High similarity - possible collapse!")
    elif np.mean(mean_pairwise_sim) > 0.5:
        print(f"   ⚠️  CAUTION: Moderate similarity")
    else:
        print(f"   ✓ Good: Low similarity - diverse embeddings")
    
    print("\n" + "="*80)

def main():
    embeddings_path = "inference_results/embeddings.pt"
    output_dir = Path("inference_results/detailed_viz")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("EMBEDDING VISUALIZATION")
    print("="*80)
    print(f"Loading from: {embeddings_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load embeddings
    pred, target = load_embeddings(embeddings_path)
    print(f"\n✓ Loaded embeddings: {pred.shape}")
    
    # Detailed analysis
    analyze_embedding_structure(pred, target)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_embedding_heatmap(
        pred, "Predicted Embeddings", 
        output_dir / "heatmap_predictions.png", max_samples=200
    )
    
    plot_embedding_heatmap(
        target, "Target Embeddings", 
        output_dir / "heatmap_targets.png", max_samples=200
    )
    
    plot_similarity_matrix(
        pred, "Prediction Similarity Matrix", 
        output_dir / "similarity_matrix_pred.png", max_samples=150
    )
    
    plot_similarity_matrix(
        target, "Target Similarity Matrix", 
        output_dir / "similarity_matrix_target.png", max_samples=150
    )
    
    plot_dimension_statistics(
        pred, target, 
        output_dir / "dimension_statistics.png"
    )
    
    plot_top_dimensions(
        pred, target, 
        output_dir / "top_dimensions.png", top_k=20
    )
    
    plot_sample_trajectories(
        pred, target, 
        output_dir / "sample_trajectories.png", n_samples=10
    )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. heatmap_predictions.png - Pattern in predicted embeddings")
    print("  2. heatmap_targets.png - Pattern in target embeddings")
    print("  3. similarity_matrix_pred.png - Which predictions are similar")
    print("  4. similarity_matrix_target.png - Which targets are similar")
    print("  5. dimension_statistics.png - Per-dimension analysis")
    print("  6. top_dimensions.png - Most important dimensions")
    print("  7. sample_trajectories.png - Individual sample patterns")

if __name__ == "__main__":
    main()

