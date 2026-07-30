from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch

from new_dt import DynamicTransformer, DynamicTransformerConfig, PackedSPRCReader


def timed(function, *, repeats: int, synchronize: bool) -> float:  # type: ignore[no-untyped-def]
    values = []
    for _ in range(repeats):
        if synchronize:
            torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        if synchronize:
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return min(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SPRC route reconstruction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    config = DynamicTransformerConfig(
        vocab_size=args.vocab,
        d_model=args.hidden,
        n_heads=4,
        n_layers=args.layers,
        ffn_dim=args.ffn,
        max_seq_len=args.seq,
        route_page_size=1024,
        route_templates_per_page=8,
        route_linear_out_tile=32,
        route_lm_head_tile=128,
        route_cache_pages=512,
    )
    model = DynamicTransformer(config).to(args.device).eval()
    token_ids = torch.randint(0, args.vocab, (args.batch, args.seq), device=args.device)
    routed = model.layers[0].ffn.up_proj.parameters_by_token

    # Warm the immutable page cache.
    routed.route_program.resolve_slice(token_ids, 0, min(routed.route_size, 8192), device=args.device)
    cache_time = timed(
        lambda: routed.route_program.resolve_slice(
            token_ids, 0, min(routed.route_size, 8192), device=args.device
        ),
        repeats=args.repeats,
        synchronize=args.device.startswith("cuda"),
    )

    with torch.no_grad():
        forward_time = timed(
            lambda: model(token_ids),
            repeats=args.repeats,
            synchronize=args.device.startswith("cuda"),
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "route.sprc"
        packed = routed.export_packed(path)
        with PackedSPRCReader(path, cache_pages=64) as reader:
            disk_time = timed(
                lambda: reader.resolve_page(0, 0),
                repeats=args.repeats,
                synchronize=False,
            )
            disk_cache = reader.cache_stats()

    storage = routed.routing_storage_estimate()
    print(f"device={args.device}")
    print(f"route_slots={routed.route_size:,}")
    print(f"estimated_packed_bytes={storage['total_bytes']:,}")
    print(f"actual_export_bytes={packed['file_bytes']:,}")
    print(f"cached_route_slice_ms={cache_time * 1000:.3f}")
    print(f"model_forward_ms={forward_time * 1000:.3f}")
    print(f"mmap_cached_page_ms={disk_time * 1000:.3f}")
    print(f"runtime_cache={routed.route_cache_stats()}")
    print(f"disk_cache={disk_cache}")


if __name__ == "__main__":
    main()
