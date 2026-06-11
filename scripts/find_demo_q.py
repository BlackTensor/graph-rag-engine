"""Find the most dramatic multi-hop win for the M9.4 demo."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eval.harness import deterministic_scores  # noqa: E402

with open("data/processed/retrieval_eval.jsonl", encoding="utf-8") as f:
    recs = [json.loads(line) for line in f if line.strip()]

for r in recs:
    r.update(deterministic_scores(r))

by_id: dict = {}
for r in recs:
    by_id.setdefault(r["id"], {})[r["pipeline"]] = r

results = []
for qid, pipes in by_id.items():
    if pipes.get("graph", {}).get("split") != "multi_hop":
        continue
    g = pipes["graph"].get("context_entity_recall", 0)
    v = pipes.get("vector", {}).get("context_entity_recall", 0)
    results.append({
        "gap": g - v,
        "graph": g,
        "vector": v,
        "id": qid,
        "question": pipes["graph"]["question"],
        "answer_entities": pipes["graph"]["answer_entities"],
        "graph_template": pipes["graph"]["graph_template"],
        "graph_contexts": pipes["graph"]["contexts"],
        "vector_contexts": pipes.get("vector", {}).get("contexts", []),
    })

results.sort(key=lambda x: x["gap"], reverse=True)

print("Top 10 multi-hop wins (gap = graph_ctx_recall - vector_ctx_recall):\n")
print(f"  {'ID':<15} {'gap':>6} {'graph':>6} {'vector':>6}  question")
print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6}  {'-'*60}")
for r in results[:10]:
    q = r["question"][:70]
    print(f"  {r['id']:<15} {r['gap']:>6.3f} {r['graph']:>6.3f} {r['vector']:>6.3f}  {q}")

best = results[0]
print("\n--- BEST CANDIDATE ---")
print(f"ID:       {best['id']}")
print(f"Template: {best['graph_template']}")
print(f"Question: {best['question']}")
print(f"Gold entities: {best['answer_entities']}")
print("\nGraph contexts (first 3):")
for c in best["graph_contexts"][:3]:
    print(f"  {c}")
print("\nVector contexts (first 3):")
for c in best["vector_contexts"][:3]:
    print(f"  {c[:100]}")
