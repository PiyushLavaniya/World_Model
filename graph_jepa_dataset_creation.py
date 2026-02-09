import argparse, json, random, os, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

##Function to parse the time
def parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None

    s = str(s).strip().replace("[UTC]", "")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    return datetime.fromisoformat(s).astimezone(timezone.utc)


def iso_time(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def is_valid_at(t: datetime, valid_at: Optional[datetime], invalid_at: Optional[datetime]) -> bool:
    if valid_at and t < valid_at:
        return False
    if invalid_at and t >= invalid_at:
        return False
    return True

def mb(n: int) -> float:
    return n / (1024 * 1024)

def approx_json_size(obj) -> int:
    # Don't call too frequently; it's expensive.
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

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


def load_nodes(path: str) -> Dict[str, NodeRec]:
    out = {}
    n_lines = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            obj = json.loads(line)
            # drop huge embeddings
            obj.pop("name_embedding", None)

            uuid = obj.get("uuid")
            labels = obj.get("labels") or []
            if isinstance(labels, str):
                labels = [labels]
            label = next((x for x in ["Episodic", "Entity", "Community"] if x in labels), "Unknown")
            out[uuid] = NodeRec(
                uuid=uuid,
                label=label,
                attrs=obj,
                created_at=parse_time(obj.get("created_at")),
                valid_at=parse_time(obj.get("valid_at")),
                invalid_at=parse_time(obj.get("invalid_at")),
            )
    print(f"[load_nodes] lines={n_lines} nodes={len(out)} time={time.time()-t0:.1f}s")
    return out

def load_edges(path: str) -> List[EdgeRec]:
    out = []
    n_lines = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            obj = json.loads(line)
            # drop huge embeddings
            obj.pop("fact_embedding", None)

            out.append(EdgeRec(
                uuid=obj.get("uuid"),
                etype=obj.get("type") or "RELATED",
                name=obj.get("name") or "UNKNOWN_RELATION",
                fact=obj.get("fact"),
                src=obj.get("source_node_uuid"),
                dst=obj.get("target_node_uuid"),
                created_at=parse_time(obj.get("created_at")),
                valid_at=parse_time(obj.get("valid_at")),
                invalid_at=parse_time(obj.get("invalid_at")),
                expired_at=parse_time(obj.get("expired_at")),
            ))
    print(f"[load_edges] lines={n_lines} edges={len(out)} time={time.time()-t0:.1f}s")
    return out


def build_snapshot(nodes: Dict[str, NodeRec], edges: List[EdgeRec], t: datetime) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for nid, n in nodes.items():
        if is_valid_at(t, n.valid_at, n.invalid_at):
            G.add_node(
                nid,
                label=n.label,
                attrs=n.attrs,
                created_at=iso_time(n.created_at) if n.created_at else None,
                valid_at=iso_time(n.valid_at) if n.valid_at else None,
                invalid_at=iso_time(n.invalid_at) if n.invalid_at else None,
            )

    for e in edges:
        if e.src not in G or e.dst not in G:
            continue
        if not is_valid_at(t, e.valid_at, e.invalid_at):
            continue
        if e.expired_at and t >= e.expired_at:
            continue
        G.add_edge(
            e.src, e.dst,
            key=e.uuid,
            uuid=e.uuid,
            etype=e.etype,
            name=e.name,
            fact=e.fact,
            created_at=iso_time(e.created_at) if e.created_at else None,
            valid_at=iso_time(e.valid_at) if e.valid_at else None,
            invalid_at=iso_time(e.invalid_at) if e.invalid_at else None,
            expired_at=iso_time(e.expired_at) if e.expired_at else None,
        )
    return G

def sample_k_hop(G: nx.MultiDiGraph, anchor: str, k: int, max_nodes: int, rng: random.Random) -> nx.MultiDiGraph:
    if anchor not in G:
        return nx.MultiDiGraph()
    visited: Set[str] = {anchor}
    frontier: List[str] = [anchor]
    for _ in range(k):
        if len(visited) >= max_nodes:
            break
        nxt = []
        rng.shuffle(frontier)
        for u in frontier:
            nbrs = list(G.successors(u)) + list(G.predecessors(u))
            rng.shuffle(nbrs)
            for v in nbrs:
                if v in visited:
                    continue
                visited.add(v)
                nxt.append(v)
                if len(visited) >= max_nodes:
                    break
            if len(visited) >= max_nodes:
                break
        frontier = nxt
        if not frontier:
            break
    return G.subgraph(visited).copy()


def split_target_nodes(SG: nx.MultiDiGraph, anchor: str, target_frac: float, rng: random.Random) -> Set[str]:
    nodes = list(SG.nodes())
    if anchor in nodes:
        nodes.remove(anchor)
    rng.shuffle(nodes)
    if not nodes:
        return set()
    n_target = max(1, int(len(nodes) * target_frac))
    return set(nodes[:n_target])


def make_views(
    SG: nx.MultiDiGraph,
    anchor: str,
    target_frac: float,
    edge_drop_prob: float,
    attr_mask_prob: float,
    keep_keys: Set[str],
    rng: random.Random
) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    target_nodes = split_target_nodes(SG, anchor, target_frac, rng)

    GT = nx.MultiDiGraph()
    for u, v, k, ed in SG.edges(keys=True, data=True):
        if u in target_nodes or v in target_nodes:
            GT.add_node(u, **SG.nodes[u])
            GT.add_node(v, **SG.nodes[v])
            GT.add_edge(u, v, key=k, **ed)

    GC = SG.copy()
    for nid in target_nodes:
        if nid in GC:
            GC.remove_node(nid)

    # edge dropout
    rm = [(u, v, k) for u, v, k in GC.edges(keys=True) if rng.random() < edge_drop_prob]
    for u, v, k in rm:
        if GC.has_edge(u, v, key=k):
            GC.remove_edge(u, v, key=k)

    # attribute masking (keep name/uuid/labels stable)
    def mask_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for kk, vv in (attrs or {}).items():
            if kk in keep_keys:
                out[kk] = vv
            else:
                if rng.random() > attr_mask_prob:
                    out[kk] = vv
        return out

    for nid in list(GC.nodes()):
        nd = GC.nodes[nid]
        nd["attrs"] = mask_attrs(nd.get("attrs", {}) or {})

    return GC, GT


def serialize_graph(G: nx.MultiDiGraph) -> Dict[str, Any]:
    nodes = []
    for nid, d in G.nodes(data=True):
        nodes.append({
            "uuid": nid,
            "label": d.get("label"),
            "attrs": d.get("attrs", {}) or {},
            "created_at": d.get("created_at"),
            "valid_at": d.get("valid_at"),
            "invalid_at": d.get("invalid_at"),
        })
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        edges.append({
            "uuid": d.get("uuid", k),
            "src": u,
            "dst": v,
            "etype": d.get("etype", "RELATED"),
            "name": d.get("name", "UNKNOWN_RELATION"),
            "fact": d.get("fact"),
            "created_at": d.get("created_at"),
            "valid_at": d.get("valid_at"),
            "invalid_at": d.get("invalid_at"),
            "expired_at": d.get("expired_at"),
        })
    return {"nodes": nodes, "edges": edges}


def canonical_target_text(GT: nx.MultiDiGraph, max_nodes: int = 30, max_edges: int = 60) -> str:
    nodes = list(GT.nodes(data=True))[:max_nodes]
    edges = list(GT.edges(keys=True, data=True))[:max_edges]
    lines = ["TARGET_SUBGRAPH", "NODES:"]
    for nid, d in nodes:
        label = d.get("label", "Unknown")
        attrs = d.get("attrs", {}) or {}
        name = str(attrs.get("name", ""))[:120]
        summary = str(attrs.get("summary", ""))[:280]
        content = str(attrs.get("content", ""))[:320]
        s = f"- {nid} | label={label}"
        if name: s += f" | name={name}"
        if label == "Entity" and summary: s += f" | summary={summary}"
        if label == "Episodic" and content: s += f" | content={content}"
        lines.append(s)
    lines.append("EDGES:")
    for u, v, k, ed in edges:
        et = ed.get("etype", "RELATED")
        rn = ed.get("name", "UNKNOWN_RELATION")
        fact = (ed.get("fact") or "")[:280]
        s = f"- ({u}) -[{et}:{rn}]-> ({v})"
        if fact: s += f" | fact={fact}"
        lines.append(s)
    return "\n".join(lines)


def pick_times(nodes: Dict[str, NodeRec], edges: List[EdgeRec], rng: random.Random, n: int) -> List[datetime]:
    cand = []
    for x in nodes.values():
        if x.valid_at: cand.append(x.valid_at)
        elif x.created_at: cand.append(x.created_at)
    for e in edges:
        if e.valid_at: cand.append(e.valid_at)
        elif e.created_at: cand.append(e.created_at)
    if not cand:
        raise ValueError("No timestamps found to sample snapshot times.")
    return [rng.choice(cand) for _ in range(n)]


def choose_anchors(G: nx.MultiDiGraph, rng: random.Random, n: int) -> List[str]:
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    pref = [x for x in nodes if G.nodes[x].get("label") in ("Episodic", "Entity")]
    rest = [x for x in nodes if x not in pref]
    nodes = pref + rest
    return nodes[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_snapshots", type=int, default=200)
    ap.add_argument("--anchors_per_snapshot", type=int, default=64)
    ap.add_argument("--k_hop", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=256)

    ap.add_argument("--target_frac", type=float, default=0.25)
    ap.add_argument("--edge_drop_prob", type=float, default=0.10)
    ap.add_argument("--attr_mask_prob", type=float, default=0.30)

    ap.add_argument("--include_target_text", action="store_true")

    # NEW: progress knobs
    ap.add_argument("--log_every", type=int, default=2000, help="print progress every N samples")
    ap.add_argument("--log_snap_every", type=int, default=10, help="print snapshot progress every N snapshots")
    ap.add_argument("--max_samples", type=int, default=0, help="cap number of samples (0 = no cap)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    t0 = time.time()
    nodes = load_nodes(args.nodes)
    edges = load_edges(args.edges)
    times = pick_times(nodes, edges, rng, args.n_snapshots)
    print(f"[diag] picked snapshot times={len(times)} load_total={time.time()-t0:.1f}s")

    keep_keys = {"uuid", "labels", "name"}

    sample_id = 0
    biggest_bytes = 0
    biggest_meta = None
    start = time.time()

    with open(args.out, "w", encoding="utf-8") as out:
        for ti, t in enumerate(times, 1):
            Gt = build_snapshot(nodes, edges, t)

            if ti == 1 or (args.log_snap_every and ti % args.log_snap_every == 0):
                print(f"[snap {ti}/{len(times)}] t={iso_time(t)} snapshot_nodes={Gt.number_of_nodes()} snapshot_edges={Gt.number_of_edges()}")
                sys.stdout.flush()

            if Gt.number_of_nodes() < 2:
                continue

            anchors = choose_anchors(Gt, rng, args.anchors_per_snapshot)
            for a in anchors:
                SG = sample_k_hop(Gt, a, args.k_hop, args.max_nodes, rng)
                if SG.number_of_nodes() < 2:
                    continue

                GC, GT = make_views(
                    SG, a, args.target_frac, args.edge_drop_prob,
                    args.attr_mask_prob, keep_keys, rng
                )
                if GT.number_of_nodes() == 0 and GT.number_of_edges() == 0:
                    continue

                rec = {
                    "sample_id": sample_id,
                    "t": iso_time(t),
                    "anchor": {"uuid": a, "label": SG.nodes[a].get("label")},
                    "context": serialize_graph(GC),
                    "target": serialize_graph(GT),
                }
                if args.include_target_text:
                    rec["target_text"] = canonical_target_text(GT)

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Progress print (occasionally)
                if args.log_every and sample_id % args.log_every == 0:
                    rec_bytes = approx_json_size(rec)
                    out.flush()
                    try:
                        os.fsync(out.fileno())
                    except Exception:
                        pass

                    out_size = os.path.getsize(args.out) if os.path.exists(args.out) else 0
                    elapsed = time.time() - start

                    if rec_bytes > biggest_bytes:
                        biggest_bytes = rec_bytes
                        biggest_meta = {
                            "sample_id": sample_id,
                            "t": iso_time(t),
                            "anchor": a,
                            "SG_nodes": SG.number_of_nodes(),
                            "SG_edges": SG.number_of_edges(),
                            "GC_nodes": len(rec["context"]["nodes"]),
                            "GC_edges": len(rec["context"]["edges"]),
                            "GT_nodes": len(rec["target"]["nodes"]),
                            "GT_edges": len(rec["target"]["edges"]),
                        }

                    print(f"[sample {sample_id}] out={mb(out_size):.1f}MB last_rec={mb(rec_bytes):.2f}MB elapsed={elapsed/60:.1f}min")
                    if biggest_meta:
                        print(f"  [biggest_rec] {mb(biggest_bytes):.2f}MB meta={biggest_meta}")
                    sys.stdout.flush()

                sample_id += 1

                if args.max_samples and sample_id >= args.max_samples:
                    print(f"[stop] reached max_samples={args.max_samples}")
                    print(f"Wrote {sample_id} samples -> {args.out}")
                    return

    print(f"Wrote {sample_id} samples -> {args.out}")


if __name__ == "__main__":
    main()
