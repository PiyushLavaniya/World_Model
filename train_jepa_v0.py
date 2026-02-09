import argparse
import json
import math
import os
import random
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import IterableDataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, MessagePassing

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time

def param_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().pow(2).sum().item())
    return math.sqrt(s)

def grad_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        s += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(s)

def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D] on CPU
    returns: [N, 2]
    """
    x = x - x.mean(dim=0, keepdim=True)
    # SVD-based PCA
    # Vh: [D, D], take top-2 right singular vectors
    _, _, Vh = torch.linalg.svd(x, full_matrices=False)
    W = Vh[:2].T  # [D, 2]
    return x @ W  # [N, 2]

def save_scatter_png(x2d: torch.Tensor, path: str, title: str = "", labels=None):
    import matplotlib
    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt

    x2d = x2d.cpu().numpy()
    plt.figure(figsize=(6, 6), dpi=150)
    if labels is None:
        plt.scatter(x2d[:, 0], x2d[:, 1], s=8, alpha=0.8)
    else:
        labels = labels.cpu().numpy()
        plt.scatter(x2d[:, 0], x2d[:, 1], s=8, alpha=0.8, c=labels)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def stable_hash(s: str) -> int:
    # stable across runs & machines
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)

def hash_bow(text: str, dim: int = 256, ngram: int = 3) -> torch.Tensor:
    """
    Simple hashing trick: character n-grams -> dense float vector.
    Fast, dependency-free, good enough for V0.
    """
    v = torch.zeros(dim, dtype=torch.float32)
    if not text:
        return v

    t = " " + " ".join(text.strip().lower().split()) + " "
    if len(t) < ngram:
        idx = stable_hash(t) % dim
        v[idx] += 1.0
        return v

    for i in range(len(t) - ngram + 1):
        g = t[i:i+ngram]
        idx = stable_hash(g) % dim
        v[idx] += 1.0

    # L2 normalize to reduce length effects
    v = v / (v.norm(p=2) + 1e-8)
    return v

def parse_iso_time(s: Optional[str]) -> Optional[float]:
    """
    Returns unix seconds (UTC) from an ISO string; tolerant to None.
    Only used for relative deltas, so seconds is enough.
    """
    if not s:
        return None
    s = str(s).strip().replace("[UTC]", "")
    # Some inputs have trailing Z; torch/python won't parse Z directly without dateutil
    # We'll do a minimal conversion: 2025-..Z -> 2025-..+00:00
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Use python datetime via torch? keep dependency-free:
    # We'll use datetime.fromisoformat
    from datetime import datetime
    dt = datetime.fromisoformat(s)
    return dt.timestamp()

LABEL2ID = {"Episodic": 0, "Entity": 1, "Community": 2, "Unknown": 3}

def node_label(node_obj: Dict) -> str:
    lbl = node_obj.get("label")
    if lbl:
        return lbl
    labels = node_obj.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    for k in ("Episodic", "Entity", "Community"):
        if k in labels:
            return k
    return "Unknown"

def make_graph_data(
    graph_obj: Dict,
    snapshot_time_iso: str,
    text_dim: int,
    time_scale_sec: float,
    rel_buckets: int,
) -> Data:
    """
    Convert {"nodes":[...], "edges":[...]} into a PyG Data object.

    Node features (float):
      - hashed BoW from name + summary + (optional clipped content)
      - time features: [age, time_to_invalidate, is_open] (scaled)

    Node label (long):
      - Episodic/Entity/Community/Unknown

    Edge rel id (long):
      - hash(etype + "::" + name) % rel_buckets
    """
    nodes = graph_obj.get("nodes", [])
    edges = graph_obj.get("edges", [])

    # uuid -> idx
    uuids = [n.get("uuid") for n in nodes]
    idx = {u: i for i, u in enumerate(uuids) if u is not None}

    t_sec = parse_iso_time(snapshot_time_iso) or 0.0

    x_list = []
    lbl_list = []

    for n in nodes:
        attrs = n.get("attrs") or {}
        name = str(attrs.get("name", "") or "")
        summary = str(attrs.get("summary", "") or "")

        # Keep content only for Episodic and clip aggressively (this avoids dataset blowups)
        lbl = node_label(n)
        content = ""
        if lbl == "Episodic":
            content = str(attrs.get("content", "") or "")[:512]

        text = (name + "\n" + summary + "\n" + content).strip()
        text_vec = hash_bow(text, dim=text_dim, ngram=3)

        valid_at = parse_iso_time(n.get("valid_at"))
        invalid_at = parse_iso_time(n.get("invalid_at"))

        # time features (scaled)
        age = 0.0
        if valid_at is not None:
            age = max(0.0, (t_sec - valid_at) / time_scale_sec)

        tti = 0.0
        if invalid_at is not None:
            tti = max(0.0, (invalid_at - t_sec) / time_scale_sec)

        is_open = 1.0 if (invalid_at is None or t_sec < invalid_at) else 0.0

        time_vec = torch.tensor([age, tti, is_open], dtype=torch.float32)

        x = torch.cat([text_vec, time_vec], dim=0)
        x_list.append(x)

        lbl_list.append(LABEL2ID.get(lbl, LABEL2ID["Unknown"]))

    if len(x_list) == 0:
        # empty graph safeguard
        x = torch.zeros((0, text_dim + 3), dtype=torch.float32)
        y = torch.zeros((0,), dtype=torch.long)
    else:
        x = torch.stack(x_list, dim=0)
        y = torch.tensor(lbl_list, dtype=torch.long)

    # edges
    srcs, dsts, rel_ids = [], [], []
    for e in edges:
        u = e.get("src")
        v = e.get("dst")
        if u not in idx or v not in idx:
            continue

        etype = str(e.get("etype", "") or "")
        name = str(e.get("name", "") or "")
        rid = stable_hash(etype + "::" + name) % rel_buckets

        srcs.append(idx[u])
        dsts.append(idx[v])
        rel_ids.append(rid)

    if len(srcs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_rel = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_rel = torch.tensor(rel_ids, dtype=torch.long)

    data = Data(x=x, label=y, edge_index=edge_index, edge_rel=edge_rel, num_nodes=x.size(0))
    return data

class JEPADatasetJSONL(IterableDataset):
    def __init__(
        self,
        path: str,
        text_dim: int = 256,
        time_scale_days: float = 30.0,
        rel_buckets: int = 4096,
        max_samples: Optional[int] = None,
        shuffle_buffer: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.path = path
        self.text_dim = text_dim
        self.time_scale_sec = time_scale_days * 24 * 3600.0
        self.rel_buckets = rel_buckets
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + int(time.time()))
        buf = []

        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples is not None and n >= self.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                t = obj.get("t") or obj.get("time") or obj.get("snapshot_time") or ""
                if not t:
                    # if missing, just use "now" placeholder
                    from datetime import datetime, timezone
                    t = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

                ctx = obj["context"]
                tgt = obj["target"]

                dc = make_graph_data(ctx, t, self.text_dim, self.time_scale_sec, self.rel_buckets)
                dt = make_graph_data(tgt, t, self.text_dim, self.time_scale_sec, self.rel_buckets)

                # skip degenerate
                if dc.num_nodes < 1 or dt.num_nodes < 1:
                    continue

                item = (dc, dt)

                if self.shuffle_buffer > 0:
                    buf.append(item)
                    if len(buf) >= self.shuffle_buffer:
                        rng.shuffle(buf)
                        while buf:
                            yield buf.pop()
                            n += 1
                            if self.max_samples is not None and n >= self.max_samples:
                                return
                else:
                    yield item
                    n += 1

        if buf:
            rng.shuffle(buf)
            for it in buf:
                yield it


def collate_pair(batch: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
    ctx_list = [b[0] for b in batch]
    tgt_list = [b[1] for b in batch]
    return Batch.from_data_list(ctx_list), Batch.from_data_list(tgt_list)

class RelSAGEConv(MessagePassing):
    """
    GraphSAGE-like conv with relation embedding injected into messages:
      m_{u->v} = W_neigh h_u + W_rel rel_emb(r)
      h_v' = W_self h_v + AGG(messages)
    This scales with many relations by hashing into rel_buckets.
    """
    def __init__(self, dim: int, rel_buckets: int, rel_dim: int = 32):
        super().__init__(aggr="mean")
        self.rel_emb = nn.Embedding(rel_buckets, rel_dim)
        self.lin_neigh = nn.Linear(dim, dim, bias=False)
        self.lin_self = nn.Linear(dim, dim, bias=True)
        self.lin_rel = nn.Linear(rel_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_rel: torch.Tensor):
        if edge_index.numel() == 0:
            return self.norm(F.relu(self.lin_self(x)))

        rel = self.rel_emb(edge_rel)  # [E, rel_dim]
        out = self.propagate(edge_index, x=x, rel=rel)
        out = self.lin_self(x) + out
        out = self.norm(F.relu(out))
        return out

    def message(self, x_j: torch.Tensor, rel: torch.Tensor):
        return self.lin_neigh(x_j) + self.lin_rel(rel)



class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        d_model: int,
        rel_buckets: int,
        n_layers: int = 2,
        label_dim: int = 32,
        rel_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(num_labels, label_dim)
        self.in_proj = nn.Linear(in_dim + label_dim, d_model)
        self.convs = nn.ModuleList([
            RelSAGEConv(d_model, rel_buckets=rel_buckets, rel_dim=rel_dim)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        # data.x: [N, in_dim], data.label: [N]
        l = self.label_emb(data.label)
        h = torch.cat([data.x, l], dim=-1)
        h = self.in_proj(h)
        h = F.relu(h)
        h = self.dropout(h)

        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_rel)
            h = self.dropout(h)

        return h  # [N, d_model]

class PredictorTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, node_h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # node_h: [N, d], batch: [N] graph id per node
        dense, mask = to_dense_batch(node_h, batch=batch)  # [B, T, d], mask [B, T]
        B, T, D = dense.shape
        cls = self.cls.expand(B, 1, D)
        x = torch.cat([cls, dense], dim=1)  # [B, 1+T, d]

        # src_key_padding_mask expects True for padding positions
        pad_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        src_key_padding_mask = ~pad_mask

        y = self.tr(x, src_key_padding_mask=src_key_padding_mask)
        y_cls = y[:, 0, :]
        return self.out(y_cls)  # [B, d]



class GraphJEPAV0(nn.Module):
    def __init__(self, in_dim: int, rel_buckets: int, d_model: int = 384):
        super().__init__()
        self.student = GraphEncoder(
            in_dim=in_dim, num_labels=len(LABEL2ID), d_model=d_model,
            rel_buckets=rel_buckets, n_layers=2
        )
        self.teacher = GraphEncoder(
            in_dim=in_dim, num_labels=len(LABEL2ID), d_model=d_model,
            rel_buckets=rel_buckets, n_layers=2
        )
        self.predictor = PredictorTransformer(d_model=d_model, n_layers=4, n_heads=6)

        # init teacher = student
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, momentum: float = 0.996):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def forward(self, ctx: Batch, tgt: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # student encodes context
        hc = self.student(ctx)                          # [Nc, d]
        pred = self.predictor(hc, ctx.batch)            # [B, d]

        # teacher encodes target (stop-grad)
        with torch.no_grad():
            ht = self.teacher(tgt)                      # [Nt, d]
            target = global_mean_pool(ht, tgt.batch)    # [B, d]
            target = F.normalize(target, dim=-1)

        pred = F.normalize(pred, dim=-1)
        return pred, target

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 1 - cosine similarity
    return (1.0 - (pred * target).sum(dim=-1)).mean()

def info_nce(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    # pred: [B,d], target: [B,d] (stop-grad)
    logits = (pred @ target.t()) / temp
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", required=True, help="dataset.jsonl produced by your creator")
#     ap.add_argument("--out_dir", default="checkpoints_v0")
#     ap.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")

#     ap.add_argument("--batch_size", type=int, default=16)
#     ap.add_argument("--num_workers", type=int, default=0)

#     ap.add_argument("--text_dim", type=int, default=256)
#     ap.add_argument("--time_scale_days", type=float, default=30.0)
#     ap.add_argument("--rel_buckets", type=int, default=4096)

#     ap.add_argument("--lr", type=float, default=2e-4)
#     ap.add_argument("--steps", type=int, default=5000)
#     ap.add_argument("--ema_m", type=float, default=0.996)

#     ap.add_argument("--log_every", type=int, default=50)
#     ap.add_argument("--save_every", type=int, default=500)

#     ap.add_argument("--shuffle_buffer", type=int, default=0, help="0 disables; else small buffer shuffle for streaming")
#     ap.add_argument("--max_samples", type=int, default=None)
#     ap.add_argument("--seed", type=int, default=42)

#     ap.add_argument("--use_infonce", action="store_true")
#     ap.add_argument("--tb_dir", default="runs/jepa_v0")
#     ap.add_argument("--embed_log_every", type=int, default=50)   # how often to log embeddings
#     ap.add_argument("--embed_k", type=int, default=128)          # how many points to log
#     ap.add_argument("--save_emb_dir", default="embeddings_v0")   # optional disk saves

#     args = ap.parse_args()
#     writer = SummaryWriter(log_dir=args.tb_dir)

#     os.makedirs(args.out_dir, exist_ok=True)
#     os.makedirs(args.save_emb_dir, exist_ok=True)

#     torch.manual_seed(args.seed)
#     random.seed(args.seed)

#     in_dim = args.text_dim + 3  # hashed text + 3 time feats

#     ds = JEPADatasetJSONL(
#         args.dataset,
#         text_dim=args.text_dim,
#         time_scale_days=args.time_scale_days,
#         rel_buckets=args.rel_buckets,
#         max_samples=args.max_samples,
#         shuffle_buffer=args.shuffle_buffer,
#         seed=args.seed,
#     )

#     # dl = DataLoader(
#     #     ds,
#     #     batch_size=args.batch_size,
#     #     num_workers=args.num_workers,
#     #     collate_fn=collate_pair,
#     # )

#     dl = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         collate_fn=collate_pair,
#         pin_memory=(args.device != "cpu"),
#         persistent_workers=(args.num_workers > 0),
#         prefetch_factor=2 if args.num_workers > 0 else None,
#     )

#     model = GraphJEPAV0(in_dim=in_dim, rel_buckets=args.rel_buckets, d_model=384).to(args.device)
#     # opt = torch.optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=0.01)
#     opt = torch.optim.AdamW(
#     list(model.student.parameters()) + list(model.predictor.parameters()),
#     lr=args.lr, weight_decay=0.01
# )

#     step = 0
#     t0 = time.time()
    
#     # streaming loader can be infinite-ish; we just break at steps
#     it = iter(dl)

#      # running stats for tqdm + prints
#     run_loss = 0.0
#     run_nodes_c = 0.0
#     run_nodes_t = 0.0
#     window_start = time.time()
#     train_start = time.time()

#     pbar = tqdm(total=args.steps, desc="train", unit="step", dynamic_ncols=True)
#     try:
#         prev_pred_norm = param_norm(model.predictor)
#         prev_student_norm = param_norm(model.student)
#         while step < args.steps:
#             try:
#                 ctx, tgt = next(it)
#             except StopIteration:
#                 it = iter(dl)
#                 ctx, tgt = next(it)

#             ctx = ctx.to(args.device)
#             tgt = tgt.to(args.device)

#             pred, target = model(ctx, tgt)

#             loss = cosine_loss(pred, target)
#             if args.use_infonce and pred.size(0) >= 2:
#                 loss = loss + info_nce(pred, target)

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             pred_gn = grad_norm(model.predictor)
#             stud_gn = grad_norm(model.student)
#             torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
#             torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)

#             opt.step()

#             new_pred_norm = param_norm(model.predictor)
#             new_student_norm = param_norm(model.student)

#             pred_delta = abs(new_pred_norm - prev_pred_norm)
#             stud_delta = abs(new_student_norm - prev_student_norm)

#             prev_pred_norm = new_pred_norm
#             prev_student_norm = new_student_norm

#             model.ema_update(momentum=args.ema_m)

#             # stats
#             run_loss += loss.item()
#             run_nodes_c += float(ctx.num_nodes)
#             run_nodes_t += float(tgt.num_nodes)

#             step += 1
#             pbar.update(1)

#             # update tqdm postfix every step (fast and useful)
#             elapsed = time.time() - train_start
#             sps = (step / elapsed) if elapsed > 0 else 0.0
#             pbar.set_postfix(
#                 loss=f"{loss.item():.4f}",
#                 ctx_n=f"{(ctx.num_nodes / max(1, ctx.num_graphs)):.1f}",
#                 tgt_n=f"{(tgt.num_nodes / max(1, tgt.num_graphs)):.1f}",
#                 sps=f"{sps:.2f}",
#                 device=str(args.device),
#             )

#             writer.add_scalar("train/loss", loss.item(), step)

#             writer.add_scalar("train/ctx_nodes_per_graph", ctx.num_nodes / max(1, ctx.num_graphs), step)
#             writer.add_scalar("train/tgt_nodes_per_graph", tgt.num_nodes / max(1, tgt.num_graphs), step)

#             writer.add_scalar("train/grad_norm/predictor", pred_gn, step)
#             writer.add_scalar("train/grad_norm/student", stud_gn, step)

#             writer.add_scalar("train/param_norm/predictor", new_pred_norm, step)
#             writer.add_scalar("train/param_norm/student", new_student_norm, step)

#             writer.add_scalar("train/param_delta/predictor", pred_delta, step)
#             writer.add_scalar("train/param_delta/student", stud_delta, step)

#             # throughput
#             elapsed = time.time() - train_start
#             writer.add_scalar("train/steps_per_sec", (step / max(1e-9, elapsed)), step)

#             if step % args.log_every == 0:
#                 dt = time.time() - window_start
#                 avg_loss = run_loss / args.log_every
#                 avg_nc = run_nodes_c / args.log_every
#                 avg_nt = run_nodes_t / args.log_every

#                 writer.add_scalar("train/loss_avg", avg_loss, step)
#                 writer.add_scalar("train/throughput_window_sps", args.log_every / dt, step)

#                 # print(
#                 #     f"[step {step}/{args.steps}] "
#                 #     f"loss={avg_loss:.4f} "
#                 #     f"avg_nodes(ctx)={avg_nc:.1f} avg_nodes(tgt)={avg_nt:.1f} "
#                 #     f"throughput={args.log_every/dt:.2f} steps/s "
#                 #     f"pred_norm={new_pred_norm:.2f} pred_Δ={pred_delta:.4e} pred_gn={pred_gn:.2e} "
#                 #     f"stud_norm={new_student_norm:.2f} stud_Δ={stud_delta:.4e} stud_gn={stud_gn:.2e} "
#                 #     f"device={args.device}"
#                 # )
#                 pbar.write(
#                     f"[step {step}/{args.steps}] loss={avg_loss:.4f} ... device={args.device}"
#                     f"pred_norm={new_pred_norm:.2f} pred_Δ={pred_delta:.4e} pred_gn={pred_gn:.2e} "
#                     f"stud_norm={new_student_norm:.2f} stud_Δ={stud_delta:.4e} stud_gn={stud_gn:.2e} "
#                 )
#                 run_loss = 0.0
#                 run_nodes_c = 0.0
#                 run_nodes_t = 0.0
#                 window_start = time.time()

#             # embeddings (only sometimes)
#             if step % args.embed_log_every == 0:
#                 with torch.no_grad():
#                     # take a small slice from current batch
#                     k = min(args.embed_k, pred.size(0))
#                     pred_k = pred[:k].detach().cpu()
#                     tgt_k  = target[:k].detach().cpu()

#                     # metadata: anchor ids (if you add them) or just “sample_i”
#                     meta = [f"step={step}_i={i}" for i in range(k)]

#                     # TensorBoard Projector (high-dim; TB can do PCA/TSNE interactively)
#                     writer.add_embedding(pred_k, metadata=meta, tag="emb/pred", global_step=step)
#                     writer.add_embedding(tgt_k,  metadata=meta, tag="emb/target", global_step=step)

#                     # Optional: save 2D PCA to disk for your own plots
#                     pred_2d = pca_2d(pred_k)
#                     tgt_2d  = pca_2d(tgt_k)

#                     save_scatter_png(
#                         pred_2d,
#                         os.path.join(args.save_emb_dir, f"pred_step_{step}.png"),
#                         title=f"pred @ step {step}",
#                     )

#                     save_scatter_png(
#                         tgt_2d,
#                         os.path.join(args.save_emb_dir, f"tgt_step_{step}.png"),
#                         title=f"target @ step {step}",
#                     )

#                     fig = plt.figure(figsize=(6,6), dpi=150)
#                     xy = pred_2d.numpy()
#                     plt.scatter(xy[:,0], xy[:,1], s=8, alpha=0.8)
#                     plt.title(f"pred @ step {step}")
#                     plt.tight_layout()
#                     writer.add_figure("emb/pred_pca2d", fig, global_step=step)
#                     plt.close(fig)

#                     fig = plt.figure(figsize=(6,6), dpi=150)
#                     xy = tgt_2d.numpy()
#                     plt.scatter(xy[:,0], xy[:,1], s=8, alpha=0.8)
#                     plt.title(f"target @ step {step}")
#                     plt.tight_layout()
#                     writer.add_figure("emb/tgt_pca2d", fig, global_step=step)
#                     plt.close(fig)

#                     torch.save(
#                         {
#                             "step": step,
#                             "pred": pred_k, "target": tgt_k,
#                             "pred_2d": pred_2d, "target_2d": tgt_2d,
#                             "meta": meta,
#                         },
#                         os.path.join(args.save_emb_dir, f"emb_step_{step}.pt"),
#                     )

#             if step % args.save_every == 0:
#                 ckpt = {
#                     "step": step,
#                     "student": model.student.state_dict(),
#                     "teacher": model.teacher.state_dict(),
#                     "predictor": model.predictor.state_dict(),
#                     "opt": opt.state_dict(),
#                     "args": vars(args),
#                 }
#                 path = os.path.join(args.out_dir, f"ckpt_step_{step}.pt")
#                 torch.save(ckpt, path)
#                 print(f"Saved checkpoint: {path}")

#     finally:
#         pbar.close()
#         writer.flush()   # optional but nice
#         writer.close()

#     # final save
#     path = os.path.join(args.out_dir, f"ckpt_final.pt")
#     torch.save(
#         {
#             "step": step,
#             "student": model.student.state_dict(),
#             "teacher": model.teacher.state_dict(),
#             "predictor": model.predictor.state_dict(),
#             "opt": opt.state_dict(),
#             "args": vars(args),
#         },
#         path,
#     )
#     print(f"Done. Saved: {path}")



# if __name__ == "__main__":
#     main()

##NEW MAIN FUNCTION WITH NEW METRICS
# --- PATCH: add these helper + metrics in your existing script ---
# (Drop-in changes; search for "### ADD:" markers)


# -----------------------------
# existing helpers...
# -----------------------------

def param_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().pow(2).sum().item())
    return math.sqrt(s)

def grad_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        s += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(s)

def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(x, full_matrices=False)
    W = Vh[:2].T
    return x @ W

def save_scatter_png(x2d: torch.Tensor, path: str, title: str = "", labels=None):
    x2d = x2d.cpu().numpy()
    plt.figure(figsize=(6, 6), dpi=150)
    if labels is None:
        plt.scatter(x2d[:, 0], x2d[:, 1], s=8, alpha=0.8)
    else:
        labels = labels.cpu().numpy()
        plt.scatter(x2d[:, 0], x2d[:, 1], s=8, alpha=0.8, c=labels)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =============================
# ### ADD: metrics helpers
# =============================

@torch.no_grad()
def batch_cos_stats(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    pred/target are expected L2-normalized [B, D]
    """
    cs = (pred * target).sum(dim=-1)  # [B]
    return float(cs.mean().item()), float(cs.std(unbiased=False).item())

@torch.no_grad()
def infonce_retrieval_stats(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> Tuple[float, float]:
    """
    Returns (top1_acc, top5_acc) for InfoNCE retrieval within the batch.
    pred/target assumed normalized.
    """
    B = pred.size(0)
    if B < 2:
        return 0.0, 0.0
    logits = (pred @ target.t()) / temp  # [B,B]
    labels = torch.arange(B, device=pred.device)

    top1 = logits.argmax(dim=1)
    top1_acc = (top1 == labels).float().mean().item()

    k = min(5, B)
    topk = logits.topk(k, dim=1).indices  # [B,k]
    top5_acc = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(top1_acc), float(top5_acc)

@torch.no_grad()
def embedding_variance_mean(x: torch.Tensor) -> float:
    """
    Collapse check: mean feature variance across batch.
    If this trends to ~0, embeddings are collapsing.
    """
    if x.numel() == 0 or x.size(0) < 2:
        return 0.0
    v = x.var(dim=0, unbiased=False).mean().item()
    return float(v)

@torch.no_grad()
def logit_diag_margins(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> Tuple[float, float]:
    """
    Another useful learning signal for InfoNCE:
      - avg positive logit (diagonal)
      - avg margin = pos - max(neg)
    """
    B = pred.size(0)
    if B < 2:
        return 0.0, 0.0
    logits = (pred @ target.t()) / temp  # [B,B]
    pos = logits.diag()                  # [B]
    # mask diagonal for negatives
    neg = logits.clone()
    neg.fill_diagonal_(-1e9)
    max_neg = neg.max(dim=1).values
    margin = (pos - max_neg)
    return float(pos.mean().item()), float(margin.mean().item())

@torch.no_grad()
def batch_metrics(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> Dict[str, float]:
    """
    pred, target: [B, D], assumed L2-normalized.
    Computes batch retrieval metrics from InfoNCE logits.
    """
    B = pred.size(0)
    out: Dict[str, float] = {}

    # cosine pairwise (positive pairs only)
    cos_pos = (pred * target).sum(dim=-1)  # [B]
    out["cos_pos_mean"] = float(cos_pos.mean().item())
    out["cos_pos_std"]  = float(cos_pos.std(unbiased=False).item())

    # variance across batch (collapse monitor)
    # mean variance per dimension
    out["var_pred"]   = float(pred.var(dim=0, unbiased=False).mean().item())
    out["var_target"] = float(target.var(dim=0, unbiased=False).mean().item())

    # norm checks (should be ~1 after normalize)
    out["pred_norm_mean"] = float(pred.norm(dim=-1).mean().item())
    out["tgt_norm_mean"]  = float(target.norm(dim=-1).mean().item())

    # InfoNCE-style logits and retrieval
    if B >= 2:
        logits = (pred @ target.t()) / temp  # [B,B]
        diag = logits.diag()                 # positive logits [B]

        # mask out diagonal to get negatives
        neg_logits = logits.clone()
        neg_logits.fill_diagonal_(-1e9)

        neg_max = neg_logits.max(dim=1).values       # hardest neg per row
        neg_mean = neg_logits.mean(dim=1)            # avg neg per row (includes -1e9? no, we replaced diag)
        # NOTE: mean includes -1e9 effect? Actually diag is -1e9 now, so mean is slightly biased.
        # Better: compute mean over off-diagonal only:
        neg_sum = (neg_logits.sum(dim=1) + 1e9)      # add back the removed diag contribution
        neg_mean_off = neg_sum / (B - 1)

        out["logit_pos_mean"] = float(diag.mean().item())
        out["logit_neg_max_mean"] = float(neg_max.mean().item())
        out["logit_neg_mean"] = float(neg_mean_off.mean().item())
        out["margin_mean"] = float((diag - neg_max).mean().item())

        # top-k accuracy (retrieve correct target index)
        # ranks: argsort descending
        # top1
        top1 = (logits.argmax(dim=1) == torch.arange(B, device=logits.device)).float().mean()
        out["top1"] = float(top1.item())
        # top5
        k = min(5, B)
        topk = logits.topk(k=k, dim=1).indices
        hit = (topk == torch.arange(B, device=logits.device).unsqueeze(1)).any(dim=1).float().mean()
        out["top5"] = float(hit.item())

    return out

# -----------------------------
# rest of your code unchanged
# -----------------------------
# stable_hash, hash_bow, dataset, model, etc...


# =============================
# ### ADD: optional args
# =============================
# inside main() argparse:
# ap.add_argument("--infonce_temp", type=float, default=0.07)

# ... in main() after args = ap.parse_args():
# (we’ll use args.infonce_temp in metrics + loss)

# -----------------------------
# Your existing info_nce (leave it, but we’ll pass temp)
# -----------------------------
def info_nce(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    logits = (pred @ target.t()) / temp
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset.jsonl produced by your creator")
    ap.add_argument("--out_dir", default="checkpoints_v0_b16")
    ap.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--time_scale_days", type=float, default=30.0)
    ap.add_argument("--rel_buckets", type=int, default=4096)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--ema_m", type=float, default=0.996)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)

    ap.add_argument("--shuffle_buffer", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_infonce", action="store_true")
    ap.add_argument("--infonce_temp", type=float, default=0.07)  # ### ADD
    ap.add_argument("--tb_dir", default="runs/jepa_v0_b16")
    ap.add_argument("--embed_log_every", type=int, default=50)
    ap.add_argument("--embed_k", type=int, default=128)
    ap.add_argument("--save_emb_dir", default="embeddings_v0_b16")
    ap.add_argument("--temp", type=float, default=1.0, help="InfoNCE temperature for metrics + loss")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from (e.g., checkpoints_v0/ckpt_step_18000.pt)")


    args = ap.parse_args()
    writer = SummaryWriter(log_dir=args.tb_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_emb_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    in_dim = args.text_dim + 3

    ds = JEPADatasetJSONL(
        args.dataset,
        text_dim=args.text_dim,
        time_scale_days=args.time_scale_days,
        rel_buckets=args.rel_buckets,
        max_samples=args.max_samples,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_pair,
        pin_memory=(args.device != "cpu"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = GraphJEPAV0(in_dim=in_dim, rel_buckets=args.rel_buckets, d_model=384).to(args.device)

    opt = torch.optim.AdamW(
        list(model.student.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    step = 0
    # Load checkpoint if resuming
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
        print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=args.device)
        model.student.load_state_dict(ckpt["student"])
        model.teacher.load_state_dict(ckpt["teacher"])
        model.predictor.load_state_dict(ckpt["predictor"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")
        # Update learning rate if it was saved in checkpoint args
        if "args" in ckpt and "lr" in ckpt["args"]:
            for param_group in opt.param_groups:
                param_group["lr"] = ckpt["args"]["lr"]
    
    it = iter(dl)

    run_loss = 0.0
    run_nodes_c = 0.0
    run_nodes_t = 0.0
    window_start = time.time()
    train_start = time.time()

    # Adjust total steps for progress bar if resuming
    remaining_steps = max(0, args.steps - step)
    pbar = tqdm(total=args.steps, initial=step, desc="train", unit="step", dynamic_ncols=True)
    try:
        prev_pred_norm = param_norm(model.predictor)
        prev_student_norm = param_norm(model.student)

        while step < args.steps:
            try:
                ctx, tgt = next(it)
            except StopIteration:
                it = iter(dl)
                ctx, tgt = next(it)

            ctx = ctx.to(args.device)
            tgt = tgt.to(args.device)

            pred, target = model(ctx, tgt)  # both are normalized already in forward()

            loss = cosine_loss(pred, target)
            if args.use_infonce and pred.size(0) >= 2:
                loss = loss + info_nce(pred, target, temp=args.infonce_temp)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            pred_gn = grad_norm(model.predictor)
            stud_gn = grad_norm(model.student)

            # ---- metrics (no_grad) ----
            m = batch_metrics(pred, target, temp=args.temp)

            # Print-friendly values
            # (top1/top5 only exist if B>=2, so use get with defaults)
            cos_mean = m["cos_pos_mean"]
            top1 = m.get("top1", float("nan"))
            top5 = m.get("top5", float("nan"))
            var_pred = m["var_pred"]
            margin = m.get("margin_mean", float("nan"))
            pos_logit = m.get("logit_pos_mean", float("nan"))
            negmax_logit = m.get("logit_neg_max_mean", float("nan"))

            torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            opt.step()

            new_pred_norm = param_norm(model.predictor)
            new_student_norm = param_norm(model.student)
            pred_delta = abs(new_pred_norm - prev_pred_norm)
            stud_delta = abs(new_student_norm - prev_student_norm)
            prev_pred_norm = new_pred_norm
            prev_student_norm = new_student_norm

            model.ema_update(momentum=args.ema_m)

            # stats
            run_loss += loss.item()
            run_nodes_c += float(ctx.num_nodes)
            run_nodes_t += float(tgt.num_nodes)

            step += 1
            pbar.update(1)

            elapsed = time.time() - train_start
            sps = (step / elapsed) if elapsed > 0 else 0.0
            # pbar.set_postfix(
            #     loss=f"{loss.item():.4f}",
            #     ctx_n=f"{(ctx.num_nodes / max(1, ctx.num_graphs)):.1f}",
            #     tgt_n=f"{(tgt.num_nodes / max(1, tgt.num_graphs)):.1f}",
            #     sps=f"{sps:.2f}",
            #     device=str(args.device),
            # )
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cos=f"{cos_mean:.3f}",
                top1=f"{top1:.3f}",
                var=f"{var_pred:.2e}",
                sps=f"{sps:.2f}",
                device=str(args.device),
            )

            # =============================
            # ### ADD: per-step metrics logs
            # =============================
            cos_mean, cos_std = batch_cos_stats(pred, target)
            pred_var = embedding_variance_mean(pred)
            tgt_var = embedding_variance_mean(target)

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/cos_sim_mean", cos_mean, step)
            writer.add_scalar("train/cos_sim_std", cos_std, step)
            writer.add_scalar("train/emb_var/pred", pred_var, step)
            writer.add_scalar("train/emb_var/target", tgt_var, step)

            if "top1" in m:
                writer.add_scalar("train/top1", m["top1"], step)
                writer.add_scalar("train/top5", m["top5"], step)
                writer.add_scalar("train/logit_pos_mean", m["logit_pos_mean"], step)
                writer.add_scalar("train/logit_neg_max_mean", m["logit_neg_max_mean"], step)
                writer.add_scalar("train/logit_neg_mean", m["logit_neg_mean"], step)
                writer.add_scalar("train/margin_mean", m["margin_mean"], step)

            if args.use_infonce and pred.size(0) >= 2:
                # top1, top5 = infonce_retrieval_stats(pred, target, temp=args.infonce_temp)
                pos_logit, margin = logit_diag_margins(pred, target, temp=args.infonce_temp)
                # writer.add_scalar("train/infonce_top1", top1, step)
                # writer.add_scalar("train/infonce_top5", top5, step)
                writer.add_scalar("train/infonce_pos_logit", pos_logit, step)
                writer.add_scalar("train/infonce_margin", margin, step)

            writer.add_scalar("train/ctx_nodes_per_graph", ctx.num_nodes / max(1, ctx.num_graphs), step)
            writer.add_scalar("train/tgt_nodes_per_graph", tgt.num_nodes / max(1, tgt.num_graphs), step)
            writer.add_scalar("train/grad_norm/predictor", pred_gn, step)
            writer.add_scalar("train/grad_norm/student", stud_gn, step)
            writer.add_scalar("train/param_norm/predictor", new_pred_norm, step)
            writer.add_scalar("train/param_norm/student", new_student_norm, step)
            writer.add_scalar("train/param_delta/predictor", pred_delta, step)
            writer.add_scalar("train/param_delta/student", stud_delta, step)
            writer.add_scalar("train/steps_per_sec", (step / max(1e-9, elapsed)), step)

            if step % args.log_every == 0:
                dt = time.time() - window_start
                avg_loss = run_loss / args.log_every
                avg_nc = run_nodes_c / args.log_every
                avg_nt = run_nodes_t / args.log_every

                writer.add_scalar("train/loss_avg", avg_loss, step)
                writer.add_scalar("train/throughput_window_sps", args.log_every / dt, step)

                # pbar.write(
                #     f"[step {step}/{args.steps}] "
                #     f"loss={avg_loss:.4f} cos={cos_mean:.3f} top1={(top1 if args.use_infonce else 0):.3f} "
                #     f"var(pred)={pred_var:.4e} "
                #     f"device={args.device}"
                # )
                pbar.write(
                    f"[step {step}/{args.steps}] "
                    f"loss={avg_loss:.4f} cos={cos_mean:.3f} "
                    f"top1={top1:.3f} top5={top5:.3f} "
                    f"margin={margin:.3f} var(pred)={var_pred:.2e} "
                    f"pos={pos_logit:.2f} negmax={negmax_logit:.2f} "
                    f"device={args.device}"
                )

                run_loss = 0.0
                run_nodes_c = 0.0
                run_nodes_t = 0.0
                window_start = time.time()

            # embeddings (your existing block stays the same)
            if step % args.embed_log_every == 0:
                with torch.no_grad():
                    k = min(args.embed_k, pred.size(0))
                    pred_k = pred[:k].detach().cpu()
                    tgt_k  = target[:k].detach().cpu()
                    meta = [f"step={step}_i={i}" for i in range(k)]

                    writer.add_embedding(pred_k, metadata=meta, tag="emb/pred", global_step=step)
                    writer.add_embedding(tgt_k,  metadata=meta, tag="emb/target", global_step=step)

                    pred_2d = pca_2d(pred_k)
                    tgt_2d  = pca_2d(tgt_k)

                    save_scatter_png(
                        pred_2d,
                        os.path.join(args.save_emb_dir, f"pred_step_{step}.png"),
                        title=f"pred @ step {step}",
                    )
                    save_scatter_png(
                        tgt_2d,
                        os.path.join(args.save_emb_dir, f"tgt_step_{step}.png"),
                        title=f"target @ step {step}",
                    )

                    fig = plt.figure(figsize=(6, 6), dpi=150)
                    xy = pred_2d.numpy()
                    plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.8)
                    plt.title(f"pred @ step {step}")
                    plt.tight_layout()
                    writer.add_figure("emb/pred_pca2d", fig, global_step=step)
                    plt.close(fig)

                    fig = plt.figure(figsize=(6, 6), dpi=150)
                    xy = tgt_2d.numpy()
                    plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.8)
                    plt.title(f"target @ step {step}")
                    plt.tight_layout()
                    writer.add_figure("emb/tgt_pca2d", fig, global_step=step)
                    plt.close(fig)

                    torch.save(
                        {"step": step, "pred": pred_k, "target": tgt_k, "pred_2d": pred_2d, "target_2d": tgt_2d, "meta": meta},
                        os.path.join(args.save_emb_dir, f"emb_step_{step}.pt"),
                    )

            if step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "student": model.student.state_dict(),
                    "teacher": model.teacher.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                }
                path = os.path.join(args.out_dir, f"ckpt_step_{step}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

    finally:
        pbar.close()
        writer.flush()
        writer.close()

    path = os.path.join(args.out_dir, f"ckpt_final.pt")
    torch.save(
        {
            "step": step,
            "student": model.student.state_dict(),
            "teacher": model.teacher.state_dict(),
            "predictor": model.predictor.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        },
        path,
    )
    print(f"Done. Saved: {path}")


if __name__ == "__main__":
    main()
