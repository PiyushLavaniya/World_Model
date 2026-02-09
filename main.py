import argparse
import json
from neo4j import GraphDatabase

NODE_LABELS = ["Episodic", "Entity", "Community"]

def _json_default(value):
    # Neo4j temporal/spatial types aren’t JSON serializable; stringify or use isoformat when available.
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        return to_native()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)

def _dumps_json(record):
    return json.dumps(record, ensure_ascii=False, default=_json_default)

def export_nodes(driver, out_path, batch_size):
    last_id = -1
    total = 0

    with driver.session() as session, open(out_path, "w", encoding = "utf-8") as out:
        while True:
            q = f"""
            MATCH (n)
            WHERE id(n) > $last_id AND any(l IN labels(n) WHERE l IN $labels)
            RETURN id(n) AS internal_id, labels(n) AS labels, properties(n) AS props
            ORDER BY id(n)
            LIMIT $batch
            """

            rows = session.run(q, last_id = last_id, labels = NODE_LABELS, batch = batch_size).data()
            if not rows:
                break

            for r in rows:
                last_id = r["internal_id"]
                props = r["props"] or {}
                # Ensure we have a stable uuid field; fall back to internal id if missing
                uuid = props.get("uuid") or props.get("id") or f"neo4j:{r['internal_id']}"
                rec = dict(props)
                rec["uuid"] = uuid
                rec["labels"] = r["labels"]
                out.write(_dumps_json(rec) + "\n")
                total += 1

    print(f"[export] nodes: {total} -> {out_path}")


def export_edges(driver, out_path: str, batch_size: int):
    last_id = -1
    total = 0
    with driver.session() as session, open(out_path, "w", encoding="utf-8") as out:
        while True:
            q = f"""
            MATCH (a)-[r]->(b)
            WHERE id(r) > $last_id
              AND any(l IN labels(a) WHERE l IN $labels)
              AND any(l IN labels(b) WHERE l IN $labels)
            RETURN id(r) AS internal_id,
                   type(r) AS rel_type,
                   properties(r) AS props,
                   coalesce(a.uuid, a.id, "neo4j:" + toString(id(a))) AS src_uuid,
                   coalesce(b.uuid, b.id, "neo4j:" + toString(id(b))) AS dst_uuid
            ORDER BY id(r)
            LIMIT $batch
            """
            rows = session.run(q, last_id=last_id, labels=NODE_LABELS, batch=batch_size).data()
            if not rows:
                break

            for r in rows:
                last_id = r["internal_id"]
                props = r["props"] or {}
                uuid = props.get("uuid") or props.get("id") or f"neo4jrel:{r['internal_id']}"

                rec = dict(props)
                rec["uuid"] = uuid
                rec["type"] = r["rel_type"]  # if you already store HAS_MEMBER/RELATES_TO/MENTIONS, it’ll appear here
                rec["source_node_uuid"] = r["src_uuid"]
                rec["target_node_uuid"] = r["dst_uuid"]
                out.write(_dumps_json(rec) + "\n")
                total += 1

    print(f"[export] edges: {total} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="neo4j+s://... or bolt://...")
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--nodes_out", default="nodes.jsonl")
    ap.add_argument("--edges_out", default="edges.jsonl")
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        export_nodes(driver, args.nodes_out, args.batch)
        export_edges(driver, args.edges_out, args.batch)
    finally:
        driver.close()

if __name__ == "__main__":
    main()
