import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from chatterbox.models.t3.train import (
    T3MedusaDistillDataset,
    collate_t3_medusa_batch,
    create_t3_medusa_model,
    describe_conditionals_file,
    describe_distill_record,
    save_medusa_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train T3 Medusa heads on distillation JSONL shards.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument("--medusa-heads", type=int, default=4)
    parser.add_argument("--medusa-layers", type=int, default=1)
    parser.add_argument("--keep-capped", action="store_true")
    parser.add_argument("--keep-dangling-quotes", action="store_true")
    parser.add_argument("--trace-shapes", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device: str):
    batch["text_tokens"] = batch["text_tokens"].to(device)
    batch["text_token_lens"] = batch["text_token_lens"].to(device)
    batch["speech_tokens"] = batch["speech_tokens"].to(device)
    batch["speech_token_lens"] = batch["speech_token_lens"].to(device)
    batch["t3_cond"] = batch["t3_cond"].to(device=device)
    return batch


@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()
    losses = []
    last_metrics = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            t3_cond=batch["t3_cond"],
            text_tokens=batch["text_tokens"],
            speech_tokens=batch["speech_tokens"],
            speech_token_lens=batch["speech_token_lens"],
            trace_shapes=False,
        )
        loss, metrics = model.compute_loss(
            medusa_logits=outputs["medusa_logits"],
            base_logits=outputs["base_logits"],
            speech_tokens=batch["speech_tokens"],
            speech_token_lens=batch["speech_token_lens"],
        )
        losses.append(loss.item())
        last_metrics = metrics
    model.train()
    if not losses:
        return {"eval_loss": math.nan}
    result = {"eval_loss": sum(losses) / len(losses)}
    for key, value in last_metrics.items():
        result[f"eval_{key}"] = value
    return result


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = T3MedusaDistillDataset(
        args.dataset_dir,
        limit=args.dataset_limit,
        drop_capped=(not args.keep_capped),
        drop_dangling_quotes=(not args.keep_dangling_quotes),
        seed=args.seed,
        shuffle=True,
    )
    print(f"dataset_dir={args.dataset_dir}")
    print(f"dataset_rows={len(dataset)}")
    print(f"dataset_stats={json.dumps(dataset.stats, ensure_ascii=False)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty after filtering")

    if args.trace_shapes:
        first_record = dataset[0]
        print(f"dataset_first_record={json.dumps(describe_distill_record(first_record), ensure_ascii=False)}")
        print(
            "dataset_first_conditionals="
            + json.dumps(
                describe_conditionals_file(first_record.conditionals_path),
                ensure_ascii=False,
            )
        )

    eval_size = 0
    if args.eval_ratio > 0 and len(dataset) >= 50:
        eval_size = max(1, int(len(dataset) * args.eval_ratio))
    train_size = len(dataset) - eval_size
    indices = list(range(len(dataset)))
    train_dataset = Subset(dataset, indices[:train_size])
    eval_dataset = Subset(dataset, indices[train_size:]) if eval_size > 0 else None

    model = create_t3_medusa_model(
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        medusa_num_heads=args.medusa_heads,
        medusa_num_layers=args.medusa_layers,
        freeze_base=True,
    )
    collate_fn = lambda rows: collate_t3_medusa_batch(rows, model.t3.hp)
    if args.trace_shapes:
        preview_rows = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
        preview_batch = collate_fn(preview_rows)
        print(
            "collate_preview="
            + json.dumps(
                {
                    "text_tokens": list(preview_batch["text_tokens"].shape),
                    "text_token_lens": preview_batch["text_token_lens"].tolist(),
                    "speech_tokens": list(preview_batch["speech_tokens"].shape),
                    "speech_token_lens": preview_batch["speech_token_lens"].tolist(),
                    "t3_cond": {
                        key: (
                            None
                            if value is None
                            else {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                            }
                        )
                        for key, value in preview_batch["t3_cond"].__dict__.items()
                    },
                },
                ensure_ascii=False,
            )
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    eval_loader = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    global_step = 0
    trace_shapes = args.trace_shapes
    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
            batch = move_batch_to_device(batch, args.device)
            outputs = model(
                t3_cond=batch["t3_cond"],
                text_tokens=batch["text_tokens"],
                speech_tokens=batch["speech_tokens"],
                speech_token_lens=batch["speech_token_lens"],
                trace_shapes=trace_shapes,
            )
            trace_shapes = False
            loss, metrics = model.compute_loss(
                medusa_logits=outputs["medusa_logits"],
                base_logits=outputs["base_logits"],
                speech_tokens=batch["speech_tokens"],
                speech_token_lens=batch["speech_token_lens"],
            )
            (loss / args.grad_accum_steps).backward()

            should_step = (batch_index + 1) % args.grad_accum_steps == 0
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        base_top1=f"{metrics['base_top1']:.3f}",
                        head0=f"{metrics.get('medusa_head_0_top1', 0.0):.3f}",
                    )

                if global_step % args.save_every == 0:
                    extra_state = {
                        "dataset_dir": args.dataset_dir,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    }
                    ckpt_dir = save_medusa_checkpoint(model, output_dir, step=global_step, extra_state=extra_state)
                    print(f"saved_checkpoint={ckpt_dir}")

                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

        if eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, args.device)
            print(f"eval_epoch_{epoch + 1}={json.dumps(eval_metrics, ensure_ascii=False)}")

    final_ckpt = save_medusa_checkpoint(
        model,
        output_dir,
        step=max(global_step, 1),
        extra_state={
            "dataset_dir": args.dataset_dir,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "global_step": global_step,
        },
    )
    print(f"final_checkpoint={final_ckpt}")


if __name__ == "__main__":
    main()
