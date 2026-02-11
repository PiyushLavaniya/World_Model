# Graph JEPA: Visual Architecture Guide

A visual, diagram-rich guide to understanding how the Graph JEPA model works.

---

## The Big Picture: What Are We Solving?

### The Challenge

**Given:** A knowledge graph that changes over time
```
                  TIME
    t=0          t=1          t=2          t=3
                                         
                                         
      
 Graph      Graph      Graph      Graph   
  @ t=0      @ t=1      @ t=2      @ t=3  
                                          
  A  B      A  B      A  B      A   B  
                                   
  C  D      C  D      C   D      C  D  
                                    
      E      E  F      E  F      E  F  
      
```

**Question:** Can we learn to predict future states from current context?

### Our Solution: Graph JEPA

**Learn to predict target graph embeddings from context graph embeddings**

```
Context Graph              Target Graph
(Partial, Corrupted)   →  (Complete, Clean)
      ↓                         ↓
   Encoder                   Encoder
      ↓                         ↓
   Embedding  PREDICT→  Embedding
   [384-dim]                [384-dim]
```

---

## End-to-End Data Flow

### Step 1: Data Extraction (Neo4j → JSONL)

```

         Neo4j Knowledge Graph                
                                              
                       
      A    B    C           
                       
                                          
                                          
                       
      D    E    F           
                       

                  ↓
         [ Export Script: main.py ]
                  ↓

         JSONL Files                          
                                              
  nodes.jsonl:                               
    {"uuid": "A", "label": "Entity", ...}    
    {"uuid": "B", "label": "Episodic", ...}  
                                              
  edges.jsonl:                               
    {"src": "A", "dst": "B", "type": ...}    

```

### Step 2: Dataset Creation (JSONL → Training Samples)

```

  Dataset Creator: graph_jepa_dataset_creation 
                                              
  FOR each temporal snapshot:                 
    FOR each anchor node:                     
      1. Extract k-hop subgraph (k=2)        
      2. Split: 75% context, 25% target      
      3. Corrupt context (drop edges, mask)   
      4. Save sample                          

                  ↓

         dataset.jsonl (9,462 samples)        
                                              
  Sample:                                     
  {                                           
    "sample_id": 42,                         
    "t": "2025-06-15T12:00:00Z",            
    "anchor": {"uuid": "A", ...},           
    "context": {                             
      "nodes": [...],  # Corrupted           
      "edges": [...]   # 10% dropped         
    },                                        
    "target": {                              
      "nodes": [...],  # Clean               
      "edges": [...]   # Complete            
    }                                         
  }                                           

```

### Step 3: Training (Dataset → Model)

```

                    Training Loop                         
                                                          
  1. Load batch of (context, target) pairs              
  2. Encode both with GNN                                
  3. Predict target embedding from context               
  4. Compute loss (cosine similarity)                    
  5. Update student & predictor                          
  6. Update teacher via EMA                              
                                                          
  Repeat 40,000 times →  Trained Model                   

```

---

## Model Architecture Deep Dive

### High-Level Architecture

```

                        GRAPH JEPA V0                          
                                                               
       
     CONTEXT PATH                TARGET PATH             
     (Trainable)                 (EMA, Frozen)           
                                                         
    Context Graph              Target Graph              
        (G_c)                      (G_t)                 
          ↓                          ↓                   
                           
       Student                  Teacher              
       Encoder                  Encoder              
       (GNN)                    (GNN)                
                           
          ↓                          ↓                   
    Node Embeddings            Node Embeddings           
     [N_c, 384]                 [N_t, 384]               
          ↓                          ↓                   
                           
     Transformer              Global Mean            
     Predictor                   Pool                
                           
          ↓                          ↓                   
    Pred Embedding             Target Embedding          
      [B, 384]                   [B, 384]                
       
              ↓                          ↓                    
              → LOSS ←                    
                     (Cosine Distance)                        

```

### Input: Graph Representation

```
Graph G:
  
  Nodes: [
    {
      uuid: "entity_123",
      label: "Entity",
      attrs: {
        name: "TechCorp",
        summary: "A technology company",
        created_at: "2024-01-01T00:00:00Z",
        valid_at: "2024-01-01T00:00:00Z"
      }
    },
    ...
  ]
  
  Edges: [
    {
      src: "entity_123",
      dst: "entity_456",
      etype: "RELATES_TO",
      name: "PROVIDES_SERVICE_TO",
      fact: "TechCorp provides cloud services to RetailCo"
    },
    ...
  ]

↓ CONVERT TO PyG Data ↓

PyG Data:
  x: [N, 259]        # Node features (text + time + label)
  edge_index: [2, E] # COO format edges
  edge_rel: [E]      # Relation type IDs
  batch: [N]         # Batch assignment
```

### Feature Engineering Pipeline

```
FOR each node:

  1. TEXT FEATURES (256-dim)
     
      text = name + summary + content      
        ↓                                  
      char_ngrams = ["tec", "ech", "chc"...] 
        ↓                                  
      hash each ngram → bucket (0-255)     
        ↓                                  
      count vector [256]                   
        ↓                                  
      L2 normalize                         
     

  2. TEMPORAL FEATURES (3-dim)
     
      age = (t_now - valid_at) / 30 days   
      tti = (invalid_at - t_now) / 30 days 
      is_open = 1 if still valid else 0    
     

  3. LABEL EMBEDDING (32-dim)
     
      label_id = {                         
        "Episodic": 0,                     
        "Entity": 1,                       
        "Community": 2,                    
        "Unknown": 3                       
      }[node.label]                        
        ↓                                  
      embedding = LookupTable[label_id]    
     

  CONCAT: [text_256 | temporal_3 | label_32] = 291-dim
  PROJECT: Linear(291 → 384)
```

### Student/Teacher Encoder (GraphSAGE + Relations)

```

               GraphEncoder                          
                                                     
  Input: x [N, 259], edge_index [2, E], edge_rel [E] 
                                                     
  1. Label Embedding                                
     label_emb = Embedding(label_id) → [N, 32]      
     x_in = concat([x, label_emb]) → [N, 291]       
                                                     
  2. Input Projection                               
     h = Linear(291 → 384) → [N, 384]               
     h = ReLU(h)                                    
     h = Dropout(h, p=0.1)                          
                                                     
  3. RelSAGEConv Layer 1                            
     h = RelSAGEConv(h, edge_index, edge_rel)       
     h = Dropout(h, p=0.1)                          
                                                     
  4. RelSAGEConv Layer 2                            
     h = RelSAGEConv(h, edge_index, edge_rel)       
     h = Dropout(h, p=0.1)                          
                                                     
  Output: h [N, 384]  (node embeddings)             

```

### RelSAGEConv: Custom Message Passing

```

            RelSAGEConv Layer                              
                                                           
  Input: x [N, 384], edge_index [2, E], edge_rel [E]      
                                                           
       
    MESSAGE PASSING                                     
                                                        
    For each edge (u → v):                              
                                                        
      1. Get source features: h_u [384]                
      2. Get relation embedding: r [32]                
         r = Embedding(edge_rel[e])                    
                                                        
      3. Compute message:                               
         m = W_neigh @ h_u + W_rel @ r                 
         (transforms: [384]→[384] and [32]→[384])      
                                                        
      4. Aggregate messages for node v:                
         agg_v = mean({m_u for all u → v})             
                                                        
       
                                                           
       
    UPDATE                                              
                                                        
    For each node v:                                    
      h_v_new = W_self @ h_v + agg_v                   
      h_v_new = LayerNorm(ReLU(h_v_new))               
                                                        
       
                                                           
  Output: h_new [N, 384]                                  

```

**Key Innovation:** Relation embeddings injected into messages, allowing the model to distinguish between different edge types.

### Predictor: Transformer Encoder

```

              Transformer Predictor                        
                                                          
  Input: node_h [N, 384], batch [N]                       
                                                          
  1. Convert to Dense Batch                              
     dense, mask = to_dense_batch(node_h, batch)         
     dense: [B, max_nodes, 384]                          
     mask: [B, max_nodes]  (True for real nodes)         
                                                          
  2. Prepend CLS Token                                   
     cls = LearnableParameter([1, 1, 384])               
     cls_expanded = cls.expand(B, 1, 384)                
     x = concat([cls_expanded, dense], dim=1)            
     x: [B, 1+max_nodes, 384]                            
                                                          
     mask_with_cls = concat([ones(B,1), mask], dim=1)    
                                                          
  3. Transformer Encoder (4 layers)                      
                 
       FOR each layer:                                 
         1. LayerNorm                                  
         2. Multi-Head Attention (6 heads)             
            (with padding mask)                         
         3. Residual connection                         
         4. LayerNorm                                  
         5. FFN (384 → 1536 → 384)                    
         6. Residual connection                         
                 
                                                          
  4. Extract CLS Embedding                               
     y_cls = x[:, 0, :]  [B, 384]                        
                                                          
  5. Output Projection                                   
     out = Linear(384 → 384)(y_cls)                      
                                                          
  6. L2 Normalize                                        
     out = out / ||out||_2                               
                                                          
  Output: pred_emb [B, 384]                              

```

**Why Transformer?**
- Self-attention captures global graph context
- CLS token aggregates information from all nodes
- Handles variable-size graphs via padding masks

### Target Encoding (Teacher Path)

```

              Teacher Encoding (EMA)                       
                                                          
  Input: target_graph (G_t)                               
                                                          
  1. Encode with Teacher GNN                             
     with torch.no_grad():  # Stop gradient!             
       h_t = TeacherEncoder(G_t)                         
       h_t: [N_t, 384]                                   
                                                          
  2. Global Mean Pooling                                 
     emb_target = global_mean_pool(h_t, batch)           
     emb_target: [B, 384]                                
                                                          
  3. L2 Normalize                                        
     emb_target = emb_target / ||emb_target||_2          
                                                          
  Output: target_emb [B, 384]                            


Note: Teacher parameters NOT updated by backprop.
      Updated via EMA from student:
      
      θ_teacher ← 0.996 × θ_teacher + 0.004 × θ_student
```

---

## Training Process Visualization

### Forward Pass

```
BATCH:
  Context Graphs (G_c): 16 graphs, various sizes
  Target Graphs (G_t): 16 graphs, various sizes

STEP 1: Student Encoding
  
    G_c → Student Encoder → h_c       
          [N_c, 384]                  
  

STEP 2: Prediction
  
    h_c → Transformer Predictor       
       → pred_emb [16, 384]           
       → L2 normalize                 
  

STEP 3: Teacher Encoding (no grad)
  
    G_t → Teacher Encoder → h_t       
          [N_t, 384]                  
       → Global Mean Pool             
       → target_emb [16, 384]         
       → L2 normalize                 
  

STEP 4: Loss Computation
  
    Cosine Loss:                      
      cos_sim = (pred ⊙ target).sum() 
      loss = 1 - cos_sim              
                                      
    Optional InfoNCE:                 
      logits = (pred @ target.T) / τ  
      loss += CrossEntropy(logits, I) 
  

STEP 5: Backward Pass
  
    loss.backward()                   
      → Gradients for student         
      → Gradients for predictor       
      → NO gradients for teacher      
  

STEP 6: Optimizer Step
  
    Clip gradients (max_norm=1.0)     
    optimizer.step()                  
      → Update student parameters     
      → Update predictor parameters   
  

STEP 7: EMA Update (Teacher)
  
    FOR each parameter:               
      θ_t ← m × θ_t + (1-m) × θ_s     
      (m = 0.996)                     
                                      
    Teacher slowly follows student    
  
```

### Learning Dynamics Over Time

```
Training Step:    0      5k     10k    15k    20k    25k    30k    35k    40k
                                                                   
Loss:          1.2 → 1.0 → 0.9 → 0.88 → 0.87 → 0.87 → 0.87 → 0.87
                                                                   
Cosine Sim:   0.60 → 0.72 → 0.78 → 0.82 → 0.84 → 0.85 → 0.86 → 0.86
                                                                   
Top-1 Acc:    0.40 → 0.55 → 0.65 → 0.70 → 0.72 → 0.73 → 0.74 → 0.75
                                                                   
Status:      [Fast Learning] [Steady Progress] [Fine-Tuning] [Convergence]
```

---

## Inference & Evaluation

### Inference Pipeline

```

                    INFERENCE PROCESS                          
                                                              
  1. Load Checkpoint                                          
      Student weights                                       
      Teacher weights                                       
      Predictor weights                                     
                                                              
  2. Set to Eval Mode                                         
     model.eval()                                             
     torch.no_grad()                                          
                                                              
  3. Load Test Dataset                                        
     dataset = JEPADatasetJSONL(test_path)                    
                                                              
  4. For Each Batch:                                          
                   
       ctx, tgt = next(dataloader)                         
       pred, target = model(ctx, tgt)                      
       metrics.update(pred, target)                        
                   
                                                              
  5. Compute Aggregate Metrics                                
      Mean/std loss                                         
      Mean/std cosine similarity                            
      Top-1/Top-5 retrieval accuracy                        
      Embedding statistics                                  
                                                              
  6. Analyze Embeddings                                       
      PCA projection                                        
      K-means clustering                                    
      Similarity distributions                              
      Nearest neighbor analysis                             
                                                              
  7. Save Results                                             
      inference_results.json                                
      embeddings.pt                                         
      visualizations/*.png                                  

```

### Evaluation Metrics Explained

```

                   EVALUATION METRICS                          
                                                               
  1. COSINE SIMILARITY                                         
                   
       cos(pred, target) = pred·target                      
                                               
                           ||pred|| ||target||               
                                                             
       Range: [-1, 1]                                       
         1.0 = Perfect alignment                             
         0.0 = Orthogonal                                   
        -1.0 = Opposite direction                           
                                                             
       Our Result: 0.861 (86.1%)                            
                   
                                                               
  2. TOP-K RETRIEVAL ACCURACY                                  
                   
       Given batch of B predictions & targets                
                                                             
       Similarity Matrix S[i,j]:                             
         S[i,j] = pred[i] · target[j]                       
                                                             
       Top-1: Does pred[i] match target[i]?                 
         accuracy = Σ(argmax_j S[i,j] == i)/B               
                                                             
       Top-5: Is target[i] in top-5 for pred[i]?            
                                                             
       Our Results:                                          
         Top-1: 74.54%                                      
         Top-5: 99.38%                                      
                   
                                                               
  3. EMBEDDING VARIANCE (Collapse Detection)                   
                   
       var = mean(var_per_dimension)                        
                                                             
       High variance (>0.01): Good diversity                 
       Low variance (<0.001): Collapse!                     
                                                             
       Our Result: ~0.0026 (healthy)                        
                   
                                                               
  4. SILHOUETTE SCORE (Cluster Quality)                        
                   
       s = (b - a) / max(a, b)                              
                                                             
       a = avg distance to same cluster                      
       b = avg distance to nearest cluster                   
                                                             
       Range: [-1, 1]                                       
         >0.5: Well-separated clusters                       
         0.2-0.5: Moderate structure                         
         <0.2: Weak/overlapping                              
                                                             
       Our Result: 0.227 (moderate, expected)                
                   

```

### Visualization Gallery

```
GENERATED VISUALIZATIONS:

1. embeddings_pca.png
   
           PCA Projection (2D)          
                                        
        •  •    •  • •                  
      •  • •  •  •   •  •               
     •    •   •      •    •             
        •   •   •  •    •               
      •       •  •   •                  
                                        
     Blue = Predictions                 
     Red = Targets                      
   

2. embeddings_clustered.png
   
         K-Means Clusters (k=5)         
                                        
      Cluster 0:                     
      Cluster 1:                     
      Cluster 2:                     
      Cluster 3:                     
      Cluster 4:                     
                                        
   

3. similarity_distributions.png
   
       Pred-Target Similarity Dist      
            ___                         
           /   \                        
          /     \                       
         /       \___                   
        /            \                  
     __/              \_____            
    -1.0    0.0    0.5    1.0          
                                        
     Mean: 0.861 (strong alignment)     
   

4. heatmap_predictions.png
   
     Embedding Heatmap (200 samples)    
                                        
     Sample    384 dims    
       1                    
       2                    
      ...                   
      200                   
                                        
     Rich activation patterns across    
     all dimensions (no collapse)       
   
```

---

## Usage Examples

### 1. Train from Scratch

```bash
# Generate dataset
python graph_jepa_dataset_creation.py \
    --nodes nodes.jsonl \
    --edges edges.jsonl \
    --out dataset.jsonl \
    --n_snapshots 200 \
    --anchors_per_snapshot 64

# Train model
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --out_dir checkpoints_v0_b16 \
    --steps 40000 \
    --batch_size 16 \
    --lr 1e-4 \
    --device mps \
    --tb_dir runs/jepa_v0_b16
```

### 2. Resume Training

```bash
python train_jepa_v0.py \
    --dataset dataset.jsonl \
    --resume checkpoints_v0_b16/ckpt_step_20000.pt \
    --steps 40000 \
    --device mps
```

### 3. Run Inference

```bash
python inference.py \
    --checkpoint checkpoints_v0_b16/ckpt_final.pt \
    --dataset dataset.jsonl \
    --output_dir inference_results \
    --n_clusters 5 \
    --device mps
```

### 4. Visualize Embeddings

```bash
python visualize_embeddings.py
# Generates detailed visualizations in inference_results/detailed_viz/
```

### 5. Monitor Training

```bash
tensorboard --logdir runs/jepa_v0_b16 --port 6006
# Open http://localhost:6006 in browser
```

---

## Key Takeaways

### What Makes This Architecture Special?

1. **Self-Supervised Learning**
   - No manual labels needed
   - Learns from graph structure itself
   - Scalable to large datasets

2. **Temporal Awareness**
   - Time features integrated into nodes
   - Validity windows respected
   - Can model graph evolution

3. **Relation-Aware**
   - Thousands of edge types handled efficiently
   - Relation embeddings learned end-to-end
   - Captures semantic relationships

4. **Robust to Corruption**
   - Trained on masked, incomplete graphs
   - Learns to infer from partial information
   - Generalizes to noisy real-world data

5. **Student-Teacher Stability**
   - EMA prevents collapse
   - Stable training dynamics
   - No need for negative sampling

### When to Use This Model?

**Good for:**
- Knowledge graph embedding
- Graph similarity search
- Link prediction
- Temporal forecasting
- Transfer learning (pre-training)

**Not ideal for:**
- Node-level classification (without fine-tuning)
- Small graphs (<10 nodes)
- Static graphs (doesn't leverage temporal info)
- Real-time inference (Transformer is slow)

---

## Further Reading

**For Implementation Details:**
- See `PROJECT_SUMMARY.md` for a high-level overview
- See `DEVELOPER_GUIDE.md` for usage and code walkthroughs
- See `dataset_analysis.md` for data structure

**For Code:**
- `train_jepa_v0.py`: Training loop and model definition
- `inference.py`: Evaluation and analysis
- `graph_jepa_dataset_creation.py`: Dataset generation

**For Results:**
- `inference_results/`: Evaluation outputs
- `runs/`: TensorBoard logs
- `checkpoints_v0_b16/`: Trained models

---

*This architecture guide provides a visual, diagram-rich explanation of the Graph JEPA model. For experiments and results, see `inference_results/` and the evaluation sections in `README.md`.*

**Last Updated:** February 2026
