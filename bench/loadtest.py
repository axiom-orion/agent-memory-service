"""Load test for the memory service. Points at any base URL (local or Cloud Run)
and reports latency percentiles + throughput for /recall.

  python bench/loadtest.py --url http://localhost:8080 --seed 500 --requests 1000 --concurrency 32

Requires httpx (in the `dev` extra). The numbers reflect wherever --url points;
when pointed at a Cloud Run URL they are the deployed p50/p99.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

_QUERIES = [
    "where does the user work", "what did they say about travel plans",
    "favorite food mentioned", "who did they meet last week",
    "what is the project deadline", "recent change of address",
    "pet name", "what hobby came up", "the doctor appointment",
    "what city are they moving to",
]


async def _seed(client: httpx.AsyncClient, n: int) -> None:
    records = [{"id": f"S{i:05d}",
                "text": f"{_QUERIES[i % len(_QUERIES)]} -- note number {i}",
                "ts_day": i // 50, "importance": 0.5} for i in range(n)]
    r = await client.post("/ingest", json={"records": records}, timeout=120)
    r.raise_for_status()


async def _worker(client: httpx.AsyncClient, queue: asyncio.Queue,
                  lat: list[float], k: int) -> None:
    while True:
        try:
            q = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        start = time.perf_counter()
        resp = await client.post("/recall", json={"query": q, "k": k}, timeout=60)
        resp.raise_for_status()
        lat.append((time.perf_counter() - start) * 1000.0)
        queue.task_done()


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[idx]


async def main_async(args) -> None:
    async with httpx.AsyncClient(base_url=args.url) as client:
        h = await client.get("/healthz", timeout=120)
        h.raise_for_status()
        backend = h.json().get("embeddings_backend", "?")
        print(f"target {args.url} | embeddings_backend={backend} | seeding {args.seed} memories...")
        await _seed(client, args.seed)

        queue: asyncio.Queue = asyncio.Queue()
        for i in range(args.requests):
            queue.put_nowait(_QUERIES[i % len(_QUERIES)])
        lat: list[float] = []
        t0 = time.perf_counter()
        await asyncio.gather(*[_worker(client, queue, lat, args.k)
                               for _ in range(args.concurrency)])
        elapsed = time.perf_counter() - t0

    print(f"\n/recall  n={len(lat)}  concurrency={args.concurrency}  k={args.k}")
    print(f"  throughput : {len(lat) / elapsed:8.1f} req/s  ({elapsed:.2f}s total)")
    print(f"  mean       : {statistics.mean(lat):8.2f} ms")
    print(f"  p50        : {_pct(lat, 50):8.2f} ms")
    print(f"  p90        : {_pct(lat, 90):8.2f} ms")
    print(f"  p99        : {_pct(lat, 99):8.2f} ms")
    print(f"  max        : {max(lat):8.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--seed", type=int, default=500, help="memories to load before the run")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--k", type=int, default=5)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
