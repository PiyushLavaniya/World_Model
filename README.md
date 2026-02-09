# Graph JEPA: Self-Supervised Learning for Temporal Knowledge Graphs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A self-supervised AI model that learns to predict temporal knowledge graph dynamics using Joint-Embedding Predictive Architecture (JEPA). Achieves **86.1% cosine similarity** on graph-level prediction tasks.

---

## 🚀 Quick Start

```bash
# Run inference on pre-trained model
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --output_dir inference_results

# View results
cat inference_results/inference_results.json
open inference_results/embeddings_pca.png
```

**See [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for detailed usage.**

---

## 📊 Key Results

| Metric                      | Result                   | Interpretation                                   |
| --------------------------- | ------------------------ | ------------------------------------------------ |
| **Cosine Similarity** | **86.1% ± 11.6%** | Strong alignment between predictions and targets |
| **Top-1 Retrieval**   | **74.5%**          | Correctly retrieves target in 3 out of 4 cases   |
| **Top-5 Retrieval**   | **99.4%**          | Nearly perfect retrieval within top-5            |
| **Embedding Quality** | ✅ Diverse, no collapse  | Learned representations are rich and meaningful  |

---

## 🎯 What is Graph JEPA?

Graph JEPA is a self-supervised learning model for temporal knowledge graphs that:

1. **Learns without labels** - Uses graph structure for self-supervision
2. **Handles temporal dynamics** - Models how graphs evolve over time
3. **Robust to corruption** - Trained on masked, incomplete graphs
4. **Relation-aware** - Efficiently handles thousands of edge types

### Architecture Overview

```
Context Graph                        Target Graph
(Masked, Corrupted)                  (Clean, Complete)
       ↓                                   ↓
  Student GNN                        Teacher GNN
  (Trainable)                        (EMA, Frozen)
       ↓                                   ↓
 Transformer                          Mean Pool
  Predictor                                ↓
       ↓                            Target Embedding
Pred Embedding  ──→ COSINE LOSS ←──
  [384-dim]                         [384-dim]
```

---

## 📚 Documentation

We provide comprehensive documentation for different audiences:

| Document                                                  | Audience         | Reading Time | Purpose                           |
| --------------------------------------------------------- | ---------------- | ------------ | --------------------------------- |
| **[PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)**         | Everyone         | 5 min        | One-page presentation summary     |
| **[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)**     | Business/Product | 10 min       | Quick overview, results, roadmap  |
| **[ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md)**   | Technical        | 20 min       | Visual architecture with diagrams |
| **[RESEARCH_PAPER.md](./RESEARCH_PAPER.md)**           | Researchers      | 45 min       | Full technical paper              |
| **[DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)**         | Developers       | 30 min       | Practical code guide              |
| **[DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)** | All              | 5 min        | Navigation guide                  |

### Quick Navigation

- 🎯 **Want to understand the project?** → [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)
- 🏗️ **Want to see how it works?** → [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md)
- 📖 **Want technical details?** → [RESEARCH_PAPER.md](./RESEARCH_PAPER.md)
- 💻 **Want to use the code?** → [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)
- 🎤 **Need to present?** → [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)

---

## 🛠️ Installation

```bash
# Clone repository
git clone <repository-url>
cd W_M

# Install dependencies
pip install torch torch-geometric networkx matplotlib seaborn scikit-learn tensorboard tqdm

# Verify installation
python -c "import torch; import torch_geometric; print('✓ All dependencies installed')"
```

---

## 🎓 Usage

### Training

```bash
# Train from scratch
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --out_dir checkpoints_v0_b16 \
    --steps 40000 \
    --batch_size 16 \
    --device mps

# Resume training
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --resume checkpoints_v0_b16/ckpt_step_20000.pt \
    --steps 40000
```

### Inference

```bash
# Full evaluation
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --output_dir inference_results

# Quick test (10 batches)
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --max_batches 10
```

### Visualization

```bash
# Generate detailed visualizations
python visualize_embeddings.py

# Monitor training
tensorboard --logdir runs/jepa_v0_b16
```

**See [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for more examples.**

---

## 🏗️ Architecture

### Model Components

- **Student Encoder:** GraphSAGE with relation embeddings (2 layers, 384-dim)
- **Teacher Encoder:** EMA-updated copy of student (momentum 0.996)
- **Predictor:** Transformer (4 layers, 6 heads) for graph-level prediction
- **Total Parameters:** ~3.5M (lightweight!)

### Key Features

1. **RelSAGEConv** - Custom message passing with relation embeddings
2. **Temporal Features** - Age, validity windows, expiration flags
3. **Text Hashing** - Fast character n-gram features (256-dim)
4. **Student-Teacher** - EMA prevents representation collapse

**See [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) for detailed diagrams.**

---

## 📊 Dataset

### Structure

- **Source:** Neo4j knowledge graph
- **Samples:** 9,462 training examples
- **Node Types:** Episodic (events), Entity (objects), Community (groups)
- **Edge Types:** ~4,096 relation types (hashed)
- **Temporal:** Time-aware with validity windows

### Creating New Dataset

```bash
# 1. Export from Neo4j
python main.py \
    --uri "neo4j+s://your-instance.neo4j.io" \
    --user "neo4j" \
    --password "your_password" \
    --nodes_out nodes.jsonl \
    --edges_out edges.jsonl

# 2. Generate training samples
python graph_jepa_dataset_creation.py \
    --nodes nodes.jsonl \
    --edges edges.jsonl \
    --out dataset.jsonl \
    --n_snapshots 200 \
    --anchors_per_snapshot 64
```

**See [dataset_analysis.md](./dataset_analysis.md) for detailed structure.**

---

## 🎯 Applications

### Immediate Use Cases

1. **Graph Similarity Search** - Find similar subgraphs in knowledge base
2. **Link Prediction** - Predict missing relationships
3. **Anomaly Detection** - Identify unusual patterns
4. **Temporal Forecasting** - Predict graph evolution

### Enterprise Applications

- Customer Intelligence
- Supply Chain Modeling
- Research & Discovery
- Compliance & Risk

**See [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) for detailed applications.**

---

## 🚀 Future Roadmap

### Phase 1: Enhanced Representations (3 months)

- [ ] Replace hashing with sentence-transformers
- [ ] Scale to 100k+ samples
- [ ] Multi-modal support

### Phase 2: Natural Language Generation (6 months)

- [ ] Add GPT-style language decoder
- [ ] Train graph → text generation
- [ ] Generate subgraph explanations

### Phase 3: Interactive World Model (12 months)

- [ ] Build predictive world model
- [ ] Enable "what if" queries
- [ ] Add planning capabilities

**See [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) for detailed roadmap.**

---

## 📈 Results & Visualizations

### Training Progress (40k steps)

```
Step:      0     10k    20k    30k    40k
Loss:    1.2 ──→ 0.9 ──→ 0.88 ──→ 0.87
Cos Sim: 0.60 ──→ 0.78 ──→ 0.84 ──→ 0.86
Top-1:   0.40 ──→ 0.65 ──→ 0.72 ──→ 0.74
```

### Visualizations

- PCA projections (2D embeddings)
- Similarity matrices
- Clustering analysis (K-means)
- Embedding heatmaps
- Dimension statistics

**View results in:** `inference_results/detailed_viz/`

---

## 🔬 Technical Details

### Model Specifications

| Component                    | Details         |
| ---------------------------- | --------------- |
| **Embedding Dim**      | 384             |
| **GNN Layers**         | 2 (RelSAGEConv) |
| **Transformer Layers** | 4               |
| **Attention Heads**    | 6               |
| **Dropout**            | 0.1             |
| **Learning Rate**      | 1e-4            |
| **EMA Momentum**       | 0.996           |

### Training

- **Hardware:** Apple M-series (MPS)
- **Batch Size:** 16
- **Steps:** 40,000
- **Time:** ~8-12 hours
- **Optimizer:** AdamW (weight_decay=0.01)

**See [RESEARCH_PAPER.md](./RESEARCH_PAPER.md) Appendix A for all hyperparameters.**

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

1. **Architecture improvements** - Try different GNN layers, predictors
2. **Scaling** - Larger datasets, distributed training
3. **Applications** - Downstream tasks, real-world deployment
4. **Language generation** - Add text decoder

**See [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for development guide.**

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{graphjepa2026,
  title={Self-Supervised Graph Learning for Temporal Knowledge Graph Prediction: 
         A Joint-Embedding Predictive Architecture Approach},
  author={[Your Name]},
  year={2026},
  note={Achieves 86.1\% cosine similarity on graph-level prediction tasks}
}
```

---

## 📄 License

[Specify your license - MIT, Apache 2.0, etc.]

---

## 🙏 Acknowledgments

This work is inspired by:

- **I-JEPA** (Meta AI) - Self-supervised learning from images
- **BYOL** (DeepMind) - Bootstrap Your Own Latent
- **GraphSAGE** (Stanford) - Inductive graph learning

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation:** See [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)
- **Questions:** Open a discussion or issue

---

## 🗂️ Project Structure

```
W_M/
├── 📄 Documentation
│   ├── README.md                      # This file (landing page)
│   ├── PROJECT_SUMMARY.md             # One-page summary
│   ├── EXECUTIVE_SUMMARY.md           # Quick overview
│   ├── ARCHITECTURE_GUIDE.md          # Visual architecture
│   ├── RESEARCH_PAPER.md              # Full technical paper
│   ├── DEVELOPER_GUIDE.md             # Developer guide
│   ├── DOCUMENTATION_INDEX.md         # Navigation guide
│   └── dataset_analysis.md            # Dataset structure
│
├── 💻 Core Scripts
│   ├── main.py                        # Neo4j export
│   ├── graph_jepa_dataset_creation.py # Dataset generation
│   ├── train_jepa_v0.py              # Training
│   ├── inference.py                   # Evaluation
│   └── visualize_embeddings.py        # Analysis
│
├── 📊 Data
│   ├── nodes.jsonl                    # Raw nodes
│   ├── edges.jsonl                    # Raw edges
│   └── dataset.jsonl                  # Training data (9,462)
│
├── 💾 Models & Logs
│   ├── checkpoints_v0_b16/           # Model checkpoints
│   ├── runs/jepa_v0_b16/             # TensorBoard logs
│   ├── embeddings_v0_b16/            # Training embeddings
│   └── inference_results/            # Evaluation outputs
│
└── ⚙️ Config
    ├── pyproject.toml                 # Dependencies
    └── uv.lock                        # Lock file
```

---

## ⭐ Highlights

- 🎯 **86.1% Cosine Similarity** - Strong graph-level learning
- 🚀 **Self-Supervised** - No manual labeling needed
- ⚡ **Lightweight** - 3.5M parameters, fast inference
- 🔄 **Temporal-Aware** - Models graph evolution
- 🎨 **Well-Documented** - 5+ comprehensive guides
- 🧪 **Research-Ready** - Reproducible, extensible

---

## 🎉 Get Started!

1. **Understand:** Read [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) (10 min)
2. **Explore:** Check out [ARCHITECTURE_GUIDE.md](./ARCHITECTURE_GUIDE.md) (20 min)
3. **Run:** Follow quick start above (5 min)
4. **Experiment:** See [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for more

**Questions?** See [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md) for navigation help.

---

**Built with ❤️ for temporal knowledge graph understanding**

*Graph JEPA: Building world models that understand and explain knowledge graphs*

**Last Updated:** February 2026 | **Status:** Research prototype, active development 🚀
