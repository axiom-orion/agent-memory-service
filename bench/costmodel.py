"""Cost model for serving /recall on Cloud Run, optionally with Vertex embeddings.

  python bench/costmodel.py --latency-ms 45 --vcpu 1 --memory-gib 0.5 --vertex --chars 120

Estimates $/1,000 requests from the measured per-request latency and allocated
resources. WARNING: the unit prices below are published Cloud Run / Vertex rates
that CHANGE over time and vary by region and tier (e.g. request-based vs
instance-based billing, free tier, committed use). Verify current pricing at
https://cloud.google.com/run/pricing and https://cloud.google.com/vertex-ai/pricing
before quoting any figure.
"""
from __future__ import annotations

import argparse

# Published Cloud Run (2nd gen, request-based) rates -- VERIFY before use.
PRICE_VCPU_SEC = 0.000024     # $ per vCPU-second
PRICE_GIB_SEC = 0.0000025     # $ per GiB-second
PRICE_PER_REQUEST = 0.40 / 1_000_000  # $ per request ($0.40 / million)
# Vertex text-embedding (per 1k input characters) -- VERIFY before use.
PRICE_VERTEX_PER_1K_CHARS = 0.000025


def cost_per_1k(latency_ms: float, vcpu: float, memory_gib: float,
                vertex: bool, chars: int) -> dict:
    sec = latency_ms / 1000.0
    compute = sec * vcpu * PRICE_VCPU_SEC * 1000
    memory = sec * memory_gib * PRICE_GIB_SEC * 1000
    requests = PRICE_PER_REQUEST * 1000
    embed = (chars / 1000.0) * PRICE_VERTEX_PER_1K_CHARS * 1000 if vertex else 0.0
    total = compute + memory + requests + embed
    return {"compute": compute, "memory": memory, "requests": requests,
            "vertex_embeddings": embed, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-ms", type=float, required=True,
                    help="measured per-request wall time (use loadtest p50 or mean)")
    ap.add_argument("--vcpu", type=float, default=1.0)
    ap.add_argument("--memory-gib", type=float, default=0.5)
    ap.add_argument("--vertex", action="store_true", help="add Vertex embedding cost")
    ap.add_argument("--chars", type=int, default=100, help="chars per query (Vertex billing)")
    args = ap.parse_args()

    c = cost_per_1k(args.latency_ms, args.vcpu, args.memory_gib, args.vertex, args.chars)
    print(f"Cost per 1,000 /recall requests @ {args.latency_ms:.1f} ms, "
          f"{args.vcpu} vCPU / {args.memory_gib} GiB"
          f"{', Vertex embeddings' if args.vertex else ', local embeddings'}:\n")
    print(f"  compute (vCPU-sec)     : ${c['compute']:.5f}")
    print(f"  memory  (GiB-sec)      : ${c['memory']:.5f}")
    print(f"  requests               : ${c['requests']:.5f}")
    if args.vertex:
        print(f"  vertex embeddings      : ${c['vertex_embeddings']:.5f}")
    print(f"  {'-' * 32}")
    print(f"  TOTAL / 1k requests    : ${c['total']:.5f}")
    print(f"  => ${c['total'] / 1000:.7f} per request")
    print("\n  Unit prices are published rates that change; verify before quoting.")


if __name__ == "__main__":
    main()
