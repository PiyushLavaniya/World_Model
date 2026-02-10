# Dataset Structure Analysis: Graph JEPA Training Data

## Overview
This is a **Joint-Embedding Predictive Architecture (JEPA)** dataset for training a self-supervised graph neural network model. The model learns to predict future graph states from current context.

---

## Input: Context Graph (`context`)

**What it is:**
- A **subgraph** extracted from a larger knowledge graph at a specific timestamp
- Contains nodes and edges that represent the "current state" or "context" around an anchor node
- The context graph is **partially masked** and **corrupted** to create a learning challenge:
  - **Edge dropout**: 10% of edges are randomly removed
  - **Attribute masking**: 30% of node attributes (except `name`, `uuid`, `labels`) are masked
  - **Target nodes removed**: Nodes that will appear in the target are removed from context

**Structure:**
- **Nodes**: Entities and Episodic events with:
  - `uuid`: Unique identifier
  - `label`: "Entity" or "Episodic"
  - `attrs`: Rich attributes including:
    - `name`: Entity/event name
    - `summary`: Textual summary (for Entities)
    - `content`: Full content (for Episodic nodes, truncated to 512 chars)
    - `created_at`, `valid_at`, `invalid_at`: Temporal metadata
  - Temporal features: age, time-to-invalidation, is_open flag
  
- **Edges**: Relationships between nodes with:
  - `src`, `dst`: Source and destination node UUIDs
  - `etype`: Edge type (e.g., "RELATES_TO", "MENTIONS")
  - `name`: Relationship name (e.g., "BILLED_AS", "PROVIDES_SERVICE_TO")
  - `fact`: Optional factual description of the relationship
  - Temporal metadata: `created_at`, `valid_at`, `invalid_at`, `expired_at`

**Feature Encoding:**
- Text features: Character n-gram hashing (256-dim) of `name + summary + content`
- Temporal features: 3-dim vector (age, time-to-invalidation, is_open)
- Total node features: 259 dimensions (256 text + 3 temporal)

---

## Target: Target Graph (`target`)

**What it is:**
- A **subgraph** representing the "future state" or "complementary view" of the same knowledge graph
- Contains nodes and edges that were **removed from the context** (target_frac = 25% of nodes)
- The target graph includes:
  - All nodes selected as "target nodes" (excluding the anchor)
  - All edges connected to these target nodes (both incoming and outgoing)
  - **No corruption**: Target graph is clean, unmasked

**Purpose:**
- Represents what the model should "predict" or "reconstruct" from the context
- The model learns to predict the embedding of the target graph from the context graph

---

## Anchor Node (`anchor`)

**What it is:**
- A single node (typically an "Entity") that serves as the **starting point** for subgraph extraction
- The anchor is used to:
  - Extract a k-hop subgraph (default: 2 hops) around it
  - Ensure context and target are related (both come from the same subgraph)
- The anchor itself is **excluded from target nodes** but may appear in context

---

## Model Task: What the Model Learns

### Architecture: Graph JEPA V0

The model uses a **student-teacher architecture** with exponential moving average (EMA):

1. **Student Encoder** (trainable):
   - Takes the **context graph** as input
   - Encodes it into node embeddings using a Graph Neural Network (GNN)
   - Uses a GraphEncoder with 2 layers, 384-dim embeddings

2. **Predictor** (trainable):
   - Takes the student's node embeddings
   - Uses a Transformer (4 layers, 6 heads) to predict a graph-level embedding
   - Outputs a normalized 384-dim vector representing the predicted target

3. **Teacher Encoder** (frozen, updated via EMA):
   - Takes the **target graph** as input
   - Encodes it into node embeddings (same architecture as student)
   - Pools node embeddings to get a graph-level embedding (global mean pool)
   - Outputs a normalized 384-dim vector representing the true target

### Learning Objective

The model minimizes the **cosine distance** between:
- **Predicted embedding** (from context via student + predictor)
- **Target embedding** (from target via teacher)

**Loss Function:**
- Option 1: `cosine_loss = 1 - cosine_similarity(pred, target)`
- Option 2: `info_nce` (contrastive learning with negative samples)

### What This Achieves

1. **Self-Supervised Learning**: No labels needed - learns from graph structure itself
2. **Temporal Understanding**: Learns to predict future graph states from current context
3. **Robust Representations**: By learning to predict through corruption (masking, edge dropout), the model learns robust graph embeddings
4. **Transfer Learning**: The learned embeddings can be used for downstream tasks like:
   - Node classification
   - Link prediction
   - Graph similarity
   - Knowledge graph completion

---

## Key Insights

### 1. **Predictive Learning**
The model doesn't just learn to encode graphs - it learns to **predict future states**. This is similar to how humans understand systems: given current context, predict what might come next.

### 2. **Robustness Through Corruption**
By training on corrupted context graphs, the model learns to:
- Handle missing information
- Infer relationships from partial data
- Be robust to noise and incomplete graphs

### 3. **Temporal Awareness**
The model incorporates temporal features (age, validity windows), allowing it to:
- Understand time-dependent relationships
- Distinguish between current and historical information
- Learn temporal patterns in graph evolution

### 4. **Multi-Scale Learning**
- **Node-level**: Individual entity/event embeddings
- **Graph-level**: Global subgraph representations
- **Relationship-level**: Edge types and facts encoded in the graph structure

### 5. **Knowledge Graph Domain**
Based on the data structure, this appears to be training on a **knowledge graph** containing:
- **Entities**: Real-world entities (companies, people, locations, concepts)
- **Episodic**: Events, documents, transactions (time-bound occurrences)
- **Relationships**: Rich typed relationships with factual descriptions

The model learns to understand how knowledge graphs evolve and how entities relate to each other over time.

---

## Training Process

1. **Data Loading**: Streams from JSONL file, processes context and target graphs
2. **Graph Encoding**: Both graphs converted to PyTorch Geometric `Data` objects
3. **Forward Pass**:
   - Context → Student → Predictor → Predicted embedding
   - Target → Teacher → Pooling → Target embedding
4. **Loss Computation**: Cosine similarity loss between predictions and targets
5. **Backpropagation**: Updates student and predictor (teacher updated via EMA)
6. **Evaluation**: Embeddings logged periodically for visualization (PCA to 2D)

---

## Dataset Statistics

- **Total samples**: 9,462 entries
- **Structure**: Each entry contains:
  - `sample_id`: Integer identifier
  - `t`: ISO timestamp of the snapshot
  - `anchor`: Starting node (Entity)
  - `context`: Corrupted subgraph (input)
  - `target`: Clean target subgraph (prediction target)
  - `target_text`: Human-readable text representation (optional)

---

## Use Cases

Once trained, this model can be used for:

1. **Graph Embedding**: Generate embeddings for knowledge graph subgraphs
2. **Similarity Search**: Find similar graph structures or entities
3. **Link Prediction**: Predict missing relationships in knowledge graphs
4. **Anomaly Detection**: Identify unusual graph patterns
5. **Temporal Forecasting**: Predict how graphs will evolve over time
6. **Transfer Learning**: Fine-tune for specific downstream tasks

---

## Summary

**Input**: Corrupted, masked context subgraph (current state)  
**Target**: Clean target subgraph (future/complementary state)  
**Task**: Learn to predict target graph embeddings from context graph  
**Method**: Self-supervised learning via contrastive/predictive objective  
**Goal**: Learn robust, temporally-aware graph representations for knowledge graphs




