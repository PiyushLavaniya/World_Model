"""
Graph JEPA Inference Script

This script performs inference with a trained Graph JEPA model:
1. Loads a trained checkpoint
2. Runs inference on test data
3. Evaluates model performance
4. Provides insights from embeddings (clustering, similarity, visualization)
5. Saves results and visualizations
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm.auto import tqdm

# Import model and dataset classes from training script
import sys
sys.path.append(os.path.dirname(__file__))
from train_jepa_v0 import (
    GraphJEPAV0, JEPADatasetJSONL, collate_pair,
    pca_2d, cosine_loss, info_nce
)


class InferenceMetrics:
    """Compute and store various inference metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.cosine_sims = []
        self.top1_accs = []
        self.top5_accs = []
        self.pred_embeddings = []
        self.target_embeddings = []
        self.sample_ids = []
        self.anchor_uuids = []
        self.anchor_labels = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, 
               loss: float, metadata: Optional[Dict] = None):
        """Update metrics with batch results"""
        self.losses.append(loss)
        
        # Cosine similarity
        cos_sim = (pred * target).sum(dim=-1)  # [B]
        self.cosine_sims.extend(cos_sim.cpu().tolist())
        
        # Top-k accuracy (retrieval)
        if pred.size(0) >= 2:
            logits = pred @ target.t()  # [B, B]
            labels = torch.arange(pred.size(0), device=pred.device)
            
            # Top-1
            top1_pred = logits.argmax(dim=-1)
            top1_acc = (top1_pred == labels).float().mean().item()
            self.top1_accs.append(top1_acc)
            
            # Top-5
            if pred.size(0) >= 5:
                top5_pred = logits.topk(5, dim=-1)[1]
                top5_acc = (top5_pred == labels.unsqueeze(1)).any(dim=-1).float().mean().item()
                self.top5_accs.append(top5_acc)
        
        # Store embeddings
        self.pred_embeddings.append(pred.cpu())
        self.target_embeddings.append(target.cpu())
        
        # Store metadata if provided
        if metadata:
            self.sample_ids.extend(metadata.get("sample_ids", []))
            self.anchor_uuids.extend(metadata.get("anchor_uuids", []))
            self.anchor_labels.extend(metadata.get("anchor_labels", []))
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics"""
        metrics = {
            "loss_mean": np.mean(self.losses),
            "loss_std": np.std(self.losses),
            "cosine_sim_mean": np.mean(self.cosine_sims),
            "cosine_sim_std": np.std(self.cosine_sims),
            "cosine_sim_min": np.min(self.cosine_sims),
            "cosine_sim_max": np.max(self.cosine_sims),
        }
        
        if self.top1_accs:
            metrics["top1_accuracy"] = np.mean(self.top1_accs)
        if self.top5_accs:
            metrics["top5_accuracy"] = np.mean(self.top5_accs)
        
        return metrics
    
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all embeddings as tensors"""
        pred = torch.cat(self.pred_embeddings, dim=0)
        target = torch.cat(self.target_embeddings, dim=0)
        return pred, target


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> Tuple[GraphJEPAV0, Dict]:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    args = ckpt.get("args", {})
    step = ckpt.get("step", 0)
    
    # Reconstruct model
    text_dim = args.get("text_dim", 256)
    rel_buckets = args.get("rel_buckets", 4096)
    in_dim = text_dim + 3
    
    model = GraphJEPAV0(in_dim=in_dim, rel_buckets=rel_buckets, d_model=384)
    model.student.load_state_dict(ckpt["student"])
    model.teacher.load_state_dict(ckpt["teacher"])
    model.predictor.load_state_dict(ckpt["predictor"])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded checkpoint from step {step}")
    print(f"  Text dim: {text_dim}, Rel buckets: {rel_buckets}")
    
    return model, args


@torch.no_grad()
def run_inference(
    model: GraphJEPAV0,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    use_infonce: bool = False
) -> InferenceMetrics:
    """Run inference on dataset"""
    model.eval()
    metrics = InferenceMetrics()
    
    pbar = tqdm(dataloader, desc="Inference", total=max_batches)
    for batch_idx, (ctx, tgt) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        
        ctx = ctx.to(device)
        tgt = tgt.to(device)
        
        # Forward pass
        pred, target = model(ctx, tgt)
        
        # Compute loss
        loss = cosine_loss(pred, target)
        if use_infonce and pred.size(0) >= 2:
            loss = loss + info_nce(pred, target, temp=0.07)
        
        # Extract metadata (if available in batch)
        metadata = {
            "sample_ids": list(range(batch_idx * pred.size(0), (batch_idx + 1) * pred.size(0))),
        }
        
        metrics.update(pred, target, loss.item(), metadata)
        
        # Update progress
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cos_sim": f"{metrics.cosine_sims[-1]:.3f}" if metrics.cosine_sims else "N/A"
        })
    
    pbar.close()
    return metrics


def analyze_embeddings(
    pred: torch.Tensor,
    target: torch.Tensor,
    output_dir: str,
    n_clusters: int = 5
) -> Dict[str, any]:
    """Analyze embedding space and generate insights"""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    insights = {}
    
    # 1. Dimensionality and statistics
    print(f"\n1. Embedding Statistics:")
    print(f"   Shape: {pred_np.shape}")
    print(f"   Pred - Mean: {pred_np.mean():.4f}, Std: {pred_np.std():.4f}")
    print(f"   Target - Mean: {target_np.mean():.4f}, Std: {target_np.std():.4f}")
    
    # Compute variance per dimension
    pred_var = pred_np.var(axis=0)
    target_var = target_np.var(axis=0)
    print(f"   Pred - Avg dim variance: {pred_var.mean():.4f}")
    print(f"   Target - Avg dim variance: {target_var.mean():.4f}")
    
    insights["embedding_dim"] = pred_np.shape[1]
    insights["n_samples"] = pred_np.shape[0]
    insights["pred_mean"] = float(pred_np.mean())
    insights["pred_std"] = float(pred_np.std())
    insights["target_mean"] = float(target_np.mean())
    insights["target_std"] = float(target_np.std())
    
    # 2. PCA Analysis
    print(f"\n2. PCA Dimensionality Reduction:")
    pred_2d = pca_2d(pred)
    target_2d = pca_2d(target)
    
    # Plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(pred_2d[:, 0], pred_2d[:, 1], s=20, alpha=0.6, c='blue')
    axes[0].set_title("Predicted Embeddings (PCA)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(target_2d[:, 0], target_2d[:, 1], s=20, alpha=0.6, c='red')
    axes[1].set_title("Target Embeddings (PCA)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "embeddings_pca.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved PCA plot: {pca_path}")
    
    # 3. t-SNE Analysis (if not too many samples)
    if pred_np.shape[0] <= 2000:
        print(f"\n3. t-SNE Visualization:")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, pred_np.shape[0]-1))
        pred_tsne = tsne.fit_transform(pred_np)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, target_np.shape[0]-1))
        target_tsne = tsne.fit_transform(target_np)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].scatter(pred_tsne[:, 0], pred_tsne[:, 1], s=20, alpha=0.6, c='blue')
        axes[0].set_title("Predicted Embeddings (t-SNE)")
        axes[0].set_xlabel("Dim 1")
        axes[0].set_ylabel("Dim 2")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(target_tsne[:, 0], target_tsne[:, 1], s=20, alpha=0.6, c='red')
        axes[1].set_title("Target Embeddings (t-SNE)")
        axes[1].set_xlabel("Dim 1")
        axes[1].set_ylabel("Dim 2")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        tsne_path = os.path.join(output_dir, "embeddings_tsne.png")
        plt.savefig(tsne_path, dpi=150)
        plt.close()
        print(f"   ✓ Saved t-SNE plot: {tsne_path}")
    else:
        print(f"\n3. t-SNE Visualization: Skipped (too many samples: {pred_np.shape[0]})")
    
    # 4. Clustering Analysis
    print(f"\n4. Clustering Analysis (K-Means with k={n_clusters}):")
    kmeans_pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_clusters = kmeans_pred.fit_predict(pred_np)
    
    kmeans_target = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    target_clusters = kmeans_target.fit_predict(target_np)
    
    # Silhouette score
    if pred_np.shape[0] > n_clusters:
        pred_silhouette = silhouette_score(pred_np, pred_clusters)
        target_silhouette = silhouette_score(target_np, target_clusters)
        print(f"   Pred Silhouette Score: {pred_silhouette:.4f}")
        print(f"   Target Silhouette Score: {target_silhouette:.4f}")
        insights["pred_silhouette"] = float(pred_silhouette)
        insights["target_silhouette"] = float(target_silhouette)
    
    # Cluster distribution
    print(f"   Pred Cluster Distribution: {np.bincount(pred_clusters)}")
    print(f"   Target Cluster Distribution: {np.bincount(target_clusters)}")
    
    # Plot clusters on PCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(pred_2d[:, 0], pred_2d[:, 1], 
                               s=20, alpha=0.6, c=pred_clusters, cmap='tab10')
    axes[0].set_title(f"Predicted Embeddings (Clustered, k={n_clusters})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(target_2d[:, 0], target_2d[:, 1], 
                               s=20, alpha=0.6, c=target_clusters, cmap='tab10')
    axes[1].set_title(f"Target Embeddings (Clustered, k={n_clusters})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    cluster_path = os.path.join(output_dir, "embeddings_clustered.png")
    plt.savefig(cluster_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved cluster plot: {cluster_path}")
    
    # 5. Similarity Analysis
    print(f"\n5. Similarity Analysis:")
    
    # Pairwise cosine similarity within predictions
    pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    pred_sim_matrix = (pred_norm @ pred_norm.t()).cpu().numpy()
    
    # Remove diagonal
    mask = np.ones_like(pred_sim_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    pred_sim_off_diag = pred_sim_matrix[mask]
    
    print(f"   Pred-Pred Cosine Sim: Mean={pred_sim_off_diag.mean():.4f}, "
          f"Std={pred_sim_off_diag.std():.4f}")
    
    # Pred-Target alignment
    pred_target_sim = (pred * target).sum(dim=-1).cpu().numpy()
    print(f"   Pred-Target Cosine Sim: Mean={pred_target_sim.mean():.4f}, "
          f"Std={pred_target_sim.std():.4f}")
    
    insights["pred_pred_sim_mean"] = float(pred_sim_off_diag.mean())
    insights["pred_target_sim_mean"] = float(pred_target_sim.mean())
    
    # Plot similarity distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(pred_sim_off_diag, bins=50, alpha=0.7, color='blue')
    axes[0].set_title("Pred-Pred Cosine Similarity Distribution")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Count")
    axes[0].axvline(pred_sim_off_diag.mean(), color='red', linestyle='--', 
                    label=f'Mean: {pred_sim_off_diag.mean():.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(pred_target_sim, bins=50, alpha=0.7, color='green')
    axes[1].set_title("Pred-Target Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].axvline(pred_target_sim.mean(), color='red', linestyle='--',
                    label=f'Mean: {pred_target_sim.mean():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    sim_path = os.path.join(output_dir, "similarity_distributions.png")
    plt.savefig(sim_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved similarity plot: {sim_path}")
    
    # 6. Nearest Neighbor Analysis
    print(f"\n6. Nearest Neighbor Analysis (Top-5):")
    
    # For each prediction, find top-5 nearest targets
    sim_matrix = (pred @ target.t()).cpu().numpy()  # [N, N]
    
    # Check if predictions correctly retrieve their targets
    correct_retrievals = (sim_matrix.argmax(axis=1) == np.arange(len(sim_matrix))).sum()
    top5_correct = sum(
        np.arange(len(sim_matrix))[i] in sim_matrix[i].argsort()[-5:]
        for i in range(len(sim_matrix))
    )
    
    print(f"   Top-1 Retrieval Accuracy: {correct_retrievals / len(sim_matrix) * 100:.2f}%")
    print(f"   Top-5 Retrieval Accuracy: {top5_correct / len(sim_matrix) * 100:.2f}%")
    
    insights["top1_retrieval_acc"] = float(correct_retrievals / len(sim_matrix))
    insights["top5_retrieval_acc"] = float(top5_correct / len(sim_matrix))
    
    return insights


def print_summary(metrics: Dict, insights: Dict):
    """Print comprehensive summary"""
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    
    print("\n📊 Performance Metrics:")
    print(f"   Loss: {metrics['loss_mean']:.4f} ± {metrics['loss_std']:.4f}")
    print(f"   Cosine Similarity: {metrics['cosine_sim_mean']:.4f} ± {metrics['cosine_sim_std']:.4f}")
    print(f"   Cosine Sim Range: [{metrics['cosine_sim_min']:.4f}, {metrics['cosine_sim_max']:.4f}]")
    
    if "top1_accuracy" in metrics:
        print(f"   Top-1 Accuracy: {metrics['top1_accuracy']*100:.2f}%")
    if "top5_accuracy" in metrics:
        print(f"   Top-5 Accuracy: {metrics['top5_accuracy']*100:.2f}%")
    
    print("\n🔍 Embedding Insights:")
    print(f"   Embedding Dimension: {insights['embedding_dim']}")
    print(f"   Number of Samples: {insights['n_samples']}")
    print(f"   Pred-Pred Similarity: {insights['pred_pred_sim_mean']:.4f}")
    print(f"   Pred-Target Similarity: {insights['pred_target_sim_mean']:.4f}")
    
    if "pred_silhouette" in insights:
        print(f"   Pred Silhouette Score: {insights['pred_silhouette']:.4f}")
        print(f"   Target Silhouette Score: {insights['target_silhouette']:.4f}")
    
    print(f"   Top-1 Retrieval: {insights['top1_retrieval_acc']*100:.2f}%")
    print(f"   Top-5 Retrieval: {insights['top5_retrieval_acc']*100:.2f}%")
    
    print("\n💡 Interpretation:")
    cos_sim = metrics['cosine_sim_mean']
    if cos_sim > 0.8:
        print("   ✓ Excellent: Model predictions are very close to targets")
    elif cos_sim > 0.6:
        print("   ✓ Good: Model predictions align well with targets")
    elif cos_sim > 0.4:
        print("   ⚠ Fair: Model predictions show moderate alignment")
    else:
        print("   ✗ Poor: Model predictions need improvement")
    
    retrieval = insights['top1_retrieval_acc']
    if retrieval > 0.8:
        print("   ✓ Excellent retrieval performance")
    elif retrieval > 0.5:
        print("   ✓ Good retrieval performance")
    else:
        print("   ⚠ Retrieval performance could be improved")
    
    if "pred_silhouette" in insights:
        sil = insights['pred_silhouette']
        if sil > 0.5:
            print("   ✓ Embeddings form well-separated clusters")
        elif sil > 0.25:
            print("   ✓ Embeddings show moderate clustering structure")
        else:
            print("   ℹ Embeddings show weak clustering (expected for dense representations)")


def main():
    parser = argparse.ArgumentParser(description="Graph JEPA Inference Script")
    
    # Required args
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to test dataset.jsonl")
    
    # Optional args
    parser.add_argument("--output_dir", default="inference_results", 
                       help="Output directory for results")
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu",
                       help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Max batches to process (for quick testing)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples from dataset")
    parser.add_argument("--n_clusters", type=int, default=5,
                       help="Number of clusters for K-Means analysis")
    parser.add_argument("--use_infonce", action="store_true",
                       help="Use InfoNCE loss in addition to cosine loss")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("GRAPH JEPA INFERENCE")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # 1. Load model
    model, train_args = load_checkpoint(args.checkpoint, args.device)
    
    # 2. Create dataset
    print(f"\nLoading dataset...")
    text_dim = train_args.get("text_dim", 256)
    time_scale_days = train_args.get("time_scale_days", 30.0)
    rel_buckets = train_args.get("rel_buckets", 4096)
    
    dataset = JEPADatasetJSONL(
        args.dataset,
        text_dim=text_dim,
        time_scale_days=time_scale_days,
        rel_buckets=rel_buckets,
        max_samples=args.max_samples,
        shuffle_buffer=0,  # No shuffle for inference
        seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_pair,
        pin_memory=(args.device != "cpu"),
    )
    
    # 3. Run inference
    print(f"\nRunning inference...")
    metrics = run_inference(
        model, dataloader, args.device,
        max_batches=args.max_batches,
        use_infonce=args.use_infonce
    )
    
    # 4. Compute metrics
    print(f"\nComputing metrics...")
    metric_results = metrics.compute()
    
    # 5. Analyze embeddings
    pred_embeddings, target_embeddings = metrics.get_embeddings()
    insights = analyze_embeddings(
        pred_embeddings,
        target_embeddings,
        args.output_dir,
        n_clusters=args.n_clusters
    )
    
    # 6. Print summary
    print_summary(metric_results, insights)
    
    # 7. Save results
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "metrics": metric_results,
        "insights": insights,
        "config": {
            "batch_size": args.batch_size,
            "device": args.device,
            "n_samples": insights["n_samples"],
        }
    }
    
    results_path = os.path.join(args.output_dir, "inference_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {results_path}")
    
    # 8. Save embeddings
    embeddings_path = os.path.join(args.output_dir, "embeddings.pt")
    torch.save({
        "pred": pred_embeddings,
        "target": target_embeddings,
        "sample_ids": metrics.sample_ids,
    }, embeddings_path)
    print(f"✓ Saved embeddings to: {embeddings_path}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

