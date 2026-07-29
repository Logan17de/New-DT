from __future__ import annotations

import argparse

import torch

from new_dt import DynamicStructureController, DynamicTransformer, DynamicTransformerConfig


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str) -> torch.Tensor:
    """Synthetic next-token task: each sequence counts forward modulo vocabulary."""

    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    return (starts + offsets) % vocab_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(7)
    config = DynamicTransformerConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        max_seq_len=16,
        initial_shared_fraction=0.5,
        pool_growth_factor=1.25,
    )
    model = DynamicTransformer(config).to(args.device)
    # Incremental merge buckets assume untouched pool entries do not move.
    # Use Adam, or AdamW with weight_decay=0.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    controller = DynamicStructureController(
        structure_interval=5,
        min_owner_samples=3,
        min_gradient_magnitude=1e-7,
        min_conflict_score=0.25,
        owner_threshold_scale=0.03,
        max_splits_per_pass=2,
        enable_merge=True,
        merge_weight_tolerance=1e-6,
        merge_gradient_tolerance=1e-6,
    )

    optimizer.zero_grad(set_to_none=True)
    for optimizer_step in range(1, args.steps + 1):
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            input_ids = make_batch(4, 12, config.vocab_size, args.device)
            output = model(input_ids, labels=input_ids, collect_route_grads=True)
            assert output.loss is not None
            (output.loss / args.grad_accum).backward()
            accumulated_loss += float(output.loss.detach())
            controller.collect(model)

        optimizer.step()
        events = controller.maybe_restructure(
            model, optimizer, optimizer_step=optimizer_step
        )
        optimizer.zero_grad(set_to_none=True)

        print(
            f"step={optimizer_step:03d} "
            f"loss={accumulated_loss / args.grad_accum:.4f} "
            f"structure_events={len(events)}"
        )
        for event in events:
            print("  ", event)

    final_events = controller.maybe_restructure(
        model, optimizer, optimizer_step=args.steps, force=True
    )
    print(f"final_structure_events={len(final_events)}")
    print("pool_summary=", model.pool_summary())


if __name__ == "__main__":
    main()
