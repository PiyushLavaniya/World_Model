# Graph JEPA: Developer Guide

A practical guide for developers working with the Graph JEPA codebase.

---

## Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install torch torch-geometric networkx matplotlib seaborn scikit-learn tensorboard tqdm
# Or: pip install -r requirements.txt  (if available)
```

### 5-Minute Demo

```bash
# 1. Check that you have the dataset
ls dataset.jsonl  # Should exist with 9,462 samples

# 2. Run inference on pre-trained model
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --output_dir inference_results \
    --max_batches 10  # Quick test with 10 batches

# 3. View results
cat inference_results/inference_results.json

# 4. Check visualizations
open inference_results/embeddings_pca.png
```

---

## Codebase Structure

### File Organization

```
W_M/

  Documentation
    README.md                  # Landing page and quick start
    PROJECT_SUMMARY.md         # One-page overview
    ARCHITECTURE_GUIDE.md      # Visual architecture guide
    DEVELOPER_GUIDE.md         # This file
    dataset_analysis.md        # Dataset structure

  Core Scripts
    main.py                    # Export from Neo4j
    graph_jepa_dataset_creation.py  # Generate training samples
    train_jepa_v0.py           # Training loop
    inference.py               # Evaluation
    visualize_embeddings.py    # Analysis tools

  Data
    nodes.jsonl                # Raw nodes from Neo4j
    edges.jsonl                # Raw edges from Neo4j
    dataset.jsonl              # Training samples (9,462)

  Models & Logs
    checkpoints_v0_b16/        # Model checkpoints
    runs/jepa_v0_b16/          # TensorBoard logs
    embeddings_v0_b16/         # Training embeddings
    inference_results/        # Evaluation outputs

  Config
    pyproject.toml             # Project dependencies
```

### Key Files Explained

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | Export Neo4j → JSONL | `export_nodes()`, `export_edges()` |
| `graph_jepa_dataset_creation.py` | Create training samples | `build_snapshot()`, `make_views()`, `serialize_graph()` |
| `train_jepa_v0.py` | Model + training | `GraphJEPAV0`, `RelSAGEConv`, `PredictorTransformer`, `main()` |
| `inference.py` | Evaluation | `run_inference()`, `analyze_embeddings()` |
| `visualize_embeddings.py` | Detailed viz | `plot_embedding_heatmap()`, `analyze_embedding_structure()` |

---

## Code Deep Dive

### 1. Dataset Creation (`graph_jepa_dataset_creation.py`)

#### Key Classes/Functions

```python
# Main entry point
def main():
    """
    Generate training samples from nodes/edges JSONL.
    
    Process:
    1. Load nodes.jsonl and edges.jsonl
    2. For each temporal snapshot:
       - Build valid graph at time t
       - Choose anchor nodes
       - Extract k-hop subgraphs
       - Split into context/target
       - Corrupt context
       - Save sample
    """
    
# Data structures
@dataclass
class NodeRec:
    uuid: str
    label: str
    attrs: Dict[str, Any]
    created_at: Optional[datetime]
    valid_at: Optional[datetime]
    invalid_at: Optional[datetime]

@dataclass
class EdgeRec:
    uuid: str
    etype: str
    name: str
    fact: Optional[str]
    src: str
    dst: str
    created_at: Optional[datetime]
    valid_at: Optional[datetime]
    invalid_at: Optional[datetime]
    expired_at: Optional[datetime]

# Key functions
def build_snapshot(nodes, edges, t) -> nx.MultiDiGraph:
    """Build graph snapshot at time t with valid nodes/edges"""

def sample_k_hop(G, anchor, k, max_nodes, rng) -> nx.MultiDiGraph:
    """Extract k-hop subgraph around anchor"""

def make_views(SG, anchor, target_frac, edge_drop_prob, 
               attr_mask_prob, keep_keys, rng):
    """Split into context/target and corrupt context"""

def serialize_graph(G) -> Dict:
    """Convert NetworkX graph to JSON-serializable dict"""
```

#### Usage

```bash
python graph_jepa_dataset_creation.py \
    --nodes nodes.jsonl \
    --edges edges.jsonl \
    --out dataset.jsonl \
    --n_snapshots 200 \
    --anchors_per_snapshot 64 \
    --k_hop 2 \
    --max_nodes 256 \
    --target_frac 0.25 \
    --edge_drop_prob 0.10 \
    --attr_mask_prob 0.30
```

---

### 2. Training (`train_jepa_v0.py`)

#### Model Architecture

```python
class GraphJEPAV0(nn.Module):
    """
    Main model class.
    
    Components:
    - student: GraphEncoder (trainable)
    - teacher: GraphEncoder (EMA, frozen)
    - predictor: PredictorTransformer (trainable)
    """
    
    def __init__(self, in_dim, rel_buckets, d_model=384):
        self.student = GraphEncoder(in_dim, num_labels=4, d_model=d_model, 
                                    rel_buckets=rel_buckets, n_layers=2)
        self.teacher = GraphEncoder(in_dim, num_labels=4, d_model=d_model,
                                    rel_buckets=rel_buckets, n_layers=2)
        self.predictor = PredictorTransformer(d_model=d_model, 
                                              n_layers=4, n_heads=6)
        
        # Init teacher = student
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)
    
    def forward(self, ctx, tgt):
        """
        Args:
            ctx: Batch of context graphs
            tgt: Batch of target graphs
        
        Returns:
            pred: Predicted embeddings [B, 384]
            target: Target embeddings [B, 384]
        """
        # Student path
        hc = self.student(ctx)              # [Nc, 384]
        pred = self.predictor(hc, ctx.batch)  # [B, 384]
        pred = F.normalize(pred, dim=-1)
        
        # Teacher path (no grad)
        with torch.no_grad():
            ht = self.teacher(tgt)           # [Nt, 384]
            target = global_mean_pool(ht, tgt.batch)  # [B, 384]
            target = F.normalize(target, dim=-1)
        
        return pred, target
    
    @torch.no_grad()
    def ema_update(self, momentum=0.996):
        """Update teacher via exponential moving average"""
        for ps, pt in zip(self.student.parameters(), 
                          self.teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


class GraphEncoder(nn.Module):
    """
    GNN encoder with relation-aware message passing.
    
    Architecture:
    1. Label embedding (4 types → 32 dim)
    2. Input projection (259 → 384)
    3. 2× RelSAGEConv layers
    """
    
    def forward(self, data):
        # Embed labels
        l = self.label_emb(data.label)  # [N, 32]
        h = torch.cat([data.x, l], dim=-1)  # [N, 259+32]
        
        # Project to model dim
        h = self.in_proj(h)  # [N, 384]
        h = F.relu(h)
        h = self.dropout(h)
        
        # Apply GNN layers
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_rel)
            h = self.dropout(h)
        
        return h  # [N, 384]


class RelSAGEConv(MessagePassing):
    """
    Custom GraphSAGE with relation embeddings.
    
    Message: m_{u→v} = W_neigh h_u + W_rel rel_emb(r)
    Update: h_v' = W_self h_v + AGG(messages)
    """
    
    def forward(self, x, edge_index, edge_rel):
        if edge_index.numel() == 0:
            return self.norm(F.relu(self.lin_self(x)))
        
        rel = self.rel_emb(edge_rel)  # [E, rel_dim]
        out = self.propagate(edge_index, x=x, rel=rel)
        out = self.lin_self(x) + out
        out = self.norm(F.relu(out))
        return out
    
    def message(self, x_j, rel):
        return self.lin_neigh(x_j) + self.lin_rel(rel)


class PredictorTransformer(nn.Module):
    """
    Transformer that predicts graph embedding from node embeddings.
    
    Uses CLS token to aggregate information.
    """
    
    def forward(self, node_h, batch):
        # Convert to dense batch
        dense, mask = to_dense_batch(node_h, batch=batch)  # [B, T, D]
        B, T, D = dense.shape
        
        # Prepend CLS token
        cls = self.cls.expand(B, 1, D)
        x = torch.cat([cls, dense], dim=1)  # [B, 1+T, D]
        
        # Create padding mask
        pad_mask = torch.cat([torch.ones(B, 1, device=mask.device), mask], dim=1)
        src_key_padding_mask = ~pad_mask
        
        # Apply Transformer
        y = self.tr(x, src_key_padding_mask=src_key_padding_mask)
        y_cls = y[:, 0, :]  # Extract CLS
        
        return self.out(y_cls)  # [B, D]
```

#### Loss Functions

```python
def cosine_loss(pred, target):
    """
    Cosine distance loss.
    
    Since embeddings are normalized:
    loss = 1 - cosine_similarity
         = 1 - (pred · target)
    
    Range: [0, 2]
    - 0: Perfect alignment
    - 1: Orthogonal
    - 2: Opposite direction
    """
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def info_nce(pred, target, temp=0.07):
    """
    InfoNCE contrastive loss.
    
    Treats diagonal as positive pairs, off-diagonal as negatives.
    """
    logits = (pred @ target.t()) / temp  # [B, B]
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)
```

#### Training Loop (Simplified)

```python
def main():
    # Setup
    model = GraphJEPAV0(in_dim, rel_buckets, d_model=384).to(device)
    opt = torch.optim.AdamW(
        list(model.student.parameters()) + list(model.predictor.parameters()),
        lr=1e-4, weight_decay=0.01
    )
    
    # Load checkpoint if resuming
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.student.load_state_dict(ckpt["student"])
        model.teacher.load_state_dict(ckpt["teacher"])
        model.predictor.load_state_dict(ckpt["predictor"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
    
    # Training loop
    for step in range(start_step, max_steps):
        # Get batch
        ctx, tgt = next(dataloader)
        ctx, tgt = ctx.to(device), tgt.to(device)
        
        # Forward
        pred, target = model(ctx, tgt)
        loss = cosine_loss(pred, target)
        if use_infonce:
            loss += info_nce(pred, target, temp=0.07)
        
        # Backward
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
        opt.step()
        
        # EMA teacher
        model.ema_update(momentum=0.996)
        
        # Logging
        if step % log_every == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/cos_sim", (pred*target).sum(-1).mean(), step)
            # ... more metrics
        
        # Save checkpoint
        if step % save_every == 0:
            torch.save({
                "step": step,
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "predictor": model.predictor.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            }, f"checkpoints/ckpt_step_{step}.pt")
```

#### Usage

```bash
# Train from scratch
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --out_dir checkpoints_v0_b16 \
    --steps 40000 \
    --batch_size 16 \
    --lr 1e-4 \
    --device mps \
    --tb_dir runs/jepa_v0_b16 \
    --seed 42

# Resume training
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --resume checkpoints_v0_b16/ckpt_step_20000.pt \
    --steps 40000 \
    --device mps

# With InfoNCE loss
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --out_dir checkpoints_infonce \
    --steps 40000 \
    --use_infonce \
    --infonce_temp 0.07
```

---

### 3. Inference (`inference.py`)

#### Key Components

```python
def load_checkpoint(ckpt_path, device="cpu"):
    """Load model from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    
    # Reconstruct model
    text_dim = args.get("text_dim", 256)
    rel_buckets = args.get("rel_buckets", 4096)
    in_dim = text_dim + 3
    
    model = GraphJEPAV0(in_dim, rel_buckets, d_model=384)
    model.student.load_state_dict(ckpt["student"])
    model.teacher.load_state_dict(ckpt["teacher"])
    model.predictor.load_state_dict(ckpt["predictor"])
    model = model.to(device)
    model.eval()
    
    return model, args


@torch.no_grad()
def run_inference(model, dataloader, device, max_batches=None):
    """Run inference and collect metrics"""
    model.eval()
    metrics = InferenceMetrics()
    
    for batch_idx, (ctx, tgt) in enumerate(tqdm(dataloader)):
        if max_batches and batch_idx >= max_batches:
            break
        
        ctx, tgt = ctx.to(device), tgt.to(device)
        pred, target = model(ctx, tgt)
        loss = cosine_loss(pred, target).item()
        
        metrics.update(pred, target, loss)
    
    return metrics


def analyze_embeddings(pred, target, output_dir, n_clusters=5):
    """
    Comprehensive embedding analysis.
    
    Generates:
    - PCA projections
    - t-SNE (if <2000 samples)
    - K-means clustering
    - Similarity distributions
    - Nearest neighbor analysis
    """
    # PCA
    pred_2d = pca_2d(pred)
    target_2d = pca_2d(target)
    plot_pca(pred_2d, target_2d, output_dir)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_clusters = kmeans.fit_predict(pred.cpu().numpy())
    silhouette = silhouette_score(pred.cpu().numpy(), pred_clusters)
    
    # Similarity
    pred_norm = pred / pred.norm(dim=-1, keepdim=True)
    sim_matrix = (pred_norm @ pred_norm.t()).cpu().numpy()
    
    # ... generate plots and return insights
```

#### Usage

```bash
# Full inference
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --output_dir inference_results \
    --batch_size 16 \
    --device mps

# Quick test (10 batches)
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --max_batches 10

# Different clustering
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --n_clusters 10
```

---

### 4. Visualization (`visualize_embeddings.py`)

#### Key Functions

```python
def plot_embedding_heatmap(embeddings, title, output_path, max_samples=100):
    """
    Plot heatmap of embeddings.
    
    Shows patterns across dimensions for multiple samples.
    """
    # Sample and sort by L2 norm
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
    
    norms = np.linalg.norm(embeddings, axis=1)
    sorted_indices = np.argsort(norms)
    embeddings_sorted = embeddings[sorted_indices]
    
    # Plot
    plt.figure(figsize=(16, 10))
    plt.imshow(embeddings_sorted, aspect='auto', cmap='RdBu_r', 
               vmin=-0.3, vmax=0.3)
    plt.xlabel('Embedding Dimension (384)')
    plt.ylabel('Samples')
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path, dpi=150)


def analyze_embedding_structure(pred, target):
    """
    Print detailed statistics.
    
    Checks for:
    - Basic stats (mean, std)
    - Sparsity (near-zero values)
    - Dimension usage (active dimensions)
    - Correlation between pred/target
    - Most/least variable dimensions
    - Collapse detection (pairwise similarity)
    """
    print("EMBEDDING STRUCTURE ANALYSIS")
    print(f"Shape: {pred.shape}")
    print(f"Mean: {pred.mean():.6f}, Std: {pred.std():.6f}")
    
    # Sparsity
    sparsity = (np.abs(pred) < 0.01).sum() / pred.size * 100
    print(f"Sparsity: {sparsity:.2f}% near zero")
    
    # Active dimensions
    active = (pred.std(axis=0) > 0.01).sum()
    print(f"Active dimensions: {active}/384")
    
    # Correlation
    corr = np.array([np.corrcoef(pred[i], target[i])[0,1] for i in range(min(1000, len(pred)))])
    print(f"Pred-target correlation: {corr.mean():.4f} ± {corr.std():.4f}")
    
    # Collapse check
    pairwise_sims = []
    for _ in range(100):
        i, j = np.random.choice(len(pred), 2, replace=False)
        sim = np.dot(pred[i], pred[j]) / (np.linalg.norm(pred[i]) * np.linalg.norm(pred[j]))
        pairwise_sims.append(sim)
    
    mean_sim = np.mean(pairwise_sims)
    print(f"Mean pairwise similarity: {mean_sim:.4f}")
    if mean_sim > 0.8:
        print("WARNING: Possible collapse!")
    else:
        print("OK: Diverse embeddings")
```

#### Usage

```bash
python visualize_embeddings.py

# Output:
# - inference_results/detailed_viz/heatmap_predictions.png
# - inference_results/detailed_viz/heatmap_targets.png
# - inference_results/detailed_viz/similarity_matrix_pred.png
# - inference_results/detailed_viz/dimension_statistics.png
# - inference_results/detailed_viz/top_dimensions.png
# - inference_results/detailed_viz/sample_trajectories.png
```

---

## Common Tasks

### Task 1: Train a New Model

```bash
# 1. Ensure dataset exists
ls dataset.jsonl

# 2. Train
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --out_dir checkpoints_my_model \
    --steps 20000 \
    --batch_size 16 \
    --lr 1e-4 \
    --device mps \
    --tb_dir runs/my_model

# 3. Monitor
tensorboard --logdir runs/my_model

# 4. Evaluate
python inference.py \
    --checkpoint checkpoints_my_model/ckpt_final.pt \
    --dataset dataset.jsonl \
    --output_dir results_my_model
```

### Task 2: Resume Training from Checkpoint

```bash
# Continue from step 10k to 30k
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --resume checkpoints_v0_b16/ckpt_step_10000.pt \
    --steps 30000 \
    --device mps
```

### Task 3: Evaluate Multiple Checkpoints

```bash
# Evaluate checkpoints at different training steps
for ckpt in checkpoints_v0_b16/ckpt_step_*.pt; do
    step=$(basename $ckpt .pt | sed 's/ckpt_step_//')
    echo "Evaluating step $step"
    python inference.py \
        --checkpoint $ckpt \
        --dataset dataset.jsonl \
        --output_dir results_step_$step \
        --max_batches 50  # Quick eval
done
```

### Task 4: Generate New Dataset from Neo4j

```bash
# 1. Export from Neo4j
python main.py \
    --uri "neo4j+s://your-instance.neo4j.io" \
    --user "neo4j" \
    --password "your_password" \
    --nodes_out nodes.jsonl \
    --edges_out edges.jsonl

# 2. Create training samples
python graph_jepa_dataset_creation.py \
    --nodes nodes.jsonl \
    --edges edges.jsonl \
    --out dataset_new.jsonl \
    --n_snapshots 200 \
    --anchors_per_snapshot 64

# 3. Train on new dataset
python train_jepa_v0.py \
    --dataset dataset_new.jsonl \
    --out_dir checkpoints_new \
    --steps 40000
```

### Task 5: Analyze Specific Embeddings

```bash
# Load embeddings and analyze in Python
python
>>> import torch
>>> data = torch.load("inference_results/embeddings.pt")
>>> pred = data["pred"]  # [N, 384]
>>> target = data["target"]  # [N, 384]
>>>
>>> # Find most similar pairs
>>> sim_matrix = (pred @ target.t())
>>> i, j = sim_matrix.argmax().item() // len(pred), sim_matrix.argmax().item() % len(pred)
>>> print(f"Most similar: sample {i} to sample {j}, similarity: {sim_matrix[i,j]:.4f}")
>>>
>>> # Find outliers
>>> norms = pred.norm(dim=-1)
>>> outlier_idx = norms.argmax()
>>> print(f"Outlier sample {outlier_idx} with norm {norms[outlier_idx]:.4f}")
```

---

## Debugging Tips

### Issue 1: Training Loss Not Decreasing

**Symptoms:**
- Loss stays around 1.0-1.2
- Cosine similarity < 0.5
- Top-1 accuracy < 30%

**Possible Causes & Fixes:**

1. **Learning rate too low/high**
   ```bash
   # Try different learning rates
   python train_jepa_v0.py --lr 2e-4  # Higher
   python train_jepa_v0.py --lr 5e-5  # Lower
   ```

2. **EMA momentum wrong**
   ```bash
   # Try different momentum
   python train_jepa_v0.py --ema_m 0.99  # Faster teacher updates
   python train_jepa_v0.py --ema_m 0.999  # Slower teacher updates
   ```

3. **Batch size too small**
   ```bash
   python train_jepa_v0.py --batch_size 32  # If memory allows
   ```

4. **Gradient clipping too aggressive**
   ```python
   # In train_jepa_v0.py, adjust:
   torch.nn.utils.clip_grad_norm_(model.student.parameters(), 5.0)  # Looser
   ```

### Issue 2: Model Collapsed (High Pred-Pred Similarity)

**Symptoms:**
- All predictions very similar
- Pairwise similarity > 0.8
- Low embedding variance

**Fix:**
- Increase corruption (edge dropout, attr masking)
- Add InfoNCE loss
- Decrease EMA momentum (faster teacher)
- Check that teacher is not being trained

### Issue 3: OOM (Out of Memory)

**Symptoms:**
- CUDA/MPS out of memory error
- Process killed

**Fixes:**

1. **Reduce batch size**
   ```bash
   python train_jepa_v0.py --batch_size 8
   ```

2. **Limit max nodes in subgraphs**
   ```bash
   # In dataset creation:
   python graph_jepa_dataset_creation.py --max_nodes 128
   ```

3. **Use gradient accumulation**
   ```python
   # Modify training loop:
   accum_steps = 2
   for step in range(max_steps):
       loss = model(ctx, tgt)
       loss = loss / accum_steps
       loss.backward()
       if (step + 1) % accum_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Issue 4: Slow Training

**Symptoms:**
- <2 steps/sec on MPS
- High CPU usage

**Optimizations:**

1. **Use persistent workers**
   ```python
   # Already in code:
   dataloader = DataLoader(..., persistent_workers=True, num_workers=0)
   ```

2. **Pin memory**
   ```python
   # Already in code:
   dataloader = DataLoader(..., pin_memory=True)
   ```

3. **Reduce embedding logging**
   ```bash
   python train_jepa_v0.py --embed_log_every 500  # Less frequent
   ```

4. **Profile**
   ```python
   # Add profiling:
   with torch.profiler.profile() as prof:
       for _ in range(10):
           pred, target = model(ctx, tgt)
   print(prof.key_averages().table())
   ```

### Issue 5: Inference Results Look Wrong

**Symptoms:**
- Very low cosine similarity (<0.5)
- Random-looking embeddings

**Checks:**

1. **Model in eval mode?**
   ```python
   model.eval()  # Disable dropout
   ```

2. **Correct checkpoint?**
   ```bash
   # Check step number
   ckpt = torch.load("checkpoint.pt")
   print(ckpt["step"])
   ```

3. **Same hyperparameters?**
   ```python
   # Checkpoint stores training args
   print(ckpt["args"])
   ```

4. **Dataset same as training?**
   ```bash
   # Check dataset size
   wc -l dataset.jsonl
   ```

---

## Testing & Validation

### Unit Tests (Manual)

```python
# test_model.py
import torch
from train_jepa_v0 import GraphJEPAV0, make_graph_data

def test_model_forward():
    """Test model forward pass"""
    model = GraphJEPAV0(in_dim=259, rel_buckets=4096, d_model=384)
    
    # Dummy data
    ctx = make_dummy_graph_data(num_nodes=10, num_edges=15)
    tgt = make_dummy_graph_data(num_nodes=8, num_edges=12)
    
    pred, target = model(ctx, tgt)
    
    assert pred.shape == (1, 384)
    assert target.shape == (1, 384)
    assert torch.allclose(pred.norm(dim=-1), torch.ones(1))  # Normalized
    print("OK: Model forward pass works")

def test_ema_update():
    """Test EMA updates teacher"""
    model = GraphJEPAV0(in_dim=259, rel_buckets=4096, d_model=384)
    
    # Save initial teacher params
    teacher_param_before = model.teacher.in_proj.weight.clone()
    
    # Modify student
    model.student.in_proj.weight.data += 1.0
    
    # EMA update
    model.ema_update(momentum=0.9)
    
    # Check teacher changed
    teacher_param_after = model.teacher.in_proj.weight
    assert not torch.allclose(teacher_param_before, teacher_param_after)
    print("OK: EMA update works")

# Run tests
test_model_forward()
test_ema_update()
```

### Integration Test

```bash
# test_pipeline.sh
#!/bin/bash
set -e

echo "Testing full pipeline..."

# 1. Create tiny dataset
python graph_jepa_dataset_creation.py \
    --nodes nodes.jsonl \
    --edges edges.jsonl \
    --out test_dataset.jsonl \
    --n_snapshots 2 \
    --anchors_per_snapshot 2 \
    --max_samples 10

# 2. Train for 10 steps
python train_jepa_v0.py \
    --dataset test_dataset.jsonl \
    --out_dir test_checkpoints \
    --steps 10 \
    --batch_size 2 \
    --save_every 5

# 3. Run inference
python inference.py \
    --checkpoint test_checkpoints/ckpt_step_10.pt \
    --dataset test_dataset.jsonl \
    --output_dir test_results \
    --max_batches 2

# 4. Check outputs
if [ -f "test_results/inference_results.json" ]; then
    echo "OK: Pipeline test passed"
else
    echo "FAIL: Pipeline test failed"
    exit 1
fi

# Cleanup
rm -rf test_dataset.jsonl test_checkpoints test_results
```

---

## Performance Benchmarks

### Training Speed (Apple M1/M2 with MPS)

| Batch Size | Avg Graph Size | Steps/Sec | Memory Usage |
|------------|----------------|-----------|--------------|
| 8 | 50 nodes | 8-10 | ~6 GB |
| 16 | 50 nodes | 5-7 | ~10 GB |
| 32 | 50 nodes | 3-4 | ~16 GB (may OOM) |

### Inference Speed

| Batch Size | Graphs/Sec | Latency (ms) |
|------------|------------|--------------|
| 1 | ~15 | 65 |
| 16 | ~200 | 80 (total batch) |
| 32 | ~350 | 90 (total batch) |

### Checkpoint Sizes

| Component | Size |
|-----------|------|
| Student encoder | ~1.5 MB |
| Teacher encoder | ~1.5 MB |
| Predictor | ~1.2 MB |
| Optimizer state | ~14 MB |
| **Total checkpoint** | **~18 MB** |

---

## Next Steps

### For Researchers

1. **Experiment with architectures**
   - Try GAT instead of GraphSAGE
   - Add more GNN layers
   - Different predictor (GRU, MLP)

2. **Ablation studies**
   - Remove temporal features
   - Remove relation embeddings
   - Different corruption strategies

3. **Scale up**
   - Larger datasets
   - More model dimensions (512, 768)
   - Longer training

### For Engineers

1. **Production deployment**
   - Export to ONNX for faster inference
   - Serve via REST API (FastAPI)
   - Batch processing pipeline

2. **Monitoring**
   - Add Prometheus metrics
   - Set up alerts for training failures
   - Track data quality over time

3. **Optimization**
   - Mixed precision training (FP16)
   - Quantization for inference
   - Distributed training (multi-GPU)

### For Application Developers

1. **Build downstream tasks**
   - Link prediction API
   - Similarity search service
   - Graph classification endpoint

2. **Integrate with UI**
   - Visualize embeddings
   - Interactive graph explorer
   - Query interface

3. **Add language generation**
   - Fine-tune GPT-2/T5 on graph→text
   - Build conversational interface
   - Generate explanations

---

## Additional Resources

### Documentation
- [README.md](./README.md) - Landing page and quick start
- [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) - One-page overview
- [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) - Visual architecture
- [dataset_analysis.md](./dataset_analysis.md) - Dataset structure

### External Links
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [I-JEPA Paper](https://arxiv.org/abs/2301.08243)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started)

### Community
- Open issues on GitHub for bugs/questions
- Share results and improvements
- Contribute to documentation

---

For questions or issues, refer to the documentation or open an issue.

*Last Updated: February 2026*
