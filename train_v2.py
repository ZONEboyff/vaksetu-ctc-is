"""
train_v2.py — Training loop for the v2 model (Conv1D + BiLSTM + Attention).

Usage
-----
  python train_v2.py

Outputs
-------
  model_v2_checkpoint.pt  — best model weights (lowest val loss)
  training_log_v2.csv     — epoch-by-epoch loss history
"""

import os
import csv
import torch
import torch.nn as nn

from config import (
    BLANK_IDX,
    VOCAB,
)
from dataset_v2 import build_dataloaders_v2
from model_v2 import build_model_v2

# ── V2 training hyperparameters ──────────────────────────────────────────────
V2_NUM_EPOCHS    = 250
V2_LEARNING_RATE = 5e-4
V2_BATCH_SIZE    = 8
V2_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model_v2_checkpoint.pt")
V2_LOG_PATH      = os.path.join(os.path.dirname(__file__), "training_log_v2.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CTC greedy decode
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs, input_lengths, blank=BLANK_IDX):
    indices = log_probs.argmax(dim=-1).permute(1, 0)   # (N, T)

    decoded_batch = []
    for b in range(indices.shape[0]):
        T   = input_lengths[b].item()
        seq = indices[b, :T].tolist()

        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev:
                collapsed.append(tok)
            prev = tok
        decoded = [t for t in collapsed if t != blank]
        decoded_batch.append(decoded)

    return decoded_batch


def indices_to_glosses(indices):
    return [VOCAB.get(i, f"<{i}>") for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss  = 0.0
    num_batches = 0

    with torch.set_grad_enabled(train):
        for padded_seqs, targets, input_lengths, target_lengths in loader:
            padded_seqs    = padded_seqs.to(device)
            targets        = targets.to(device)
            input_lengths  = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(padded_seqs)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TrainV2] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders_v2(batch_size=V2_BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = build_model_v2(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=V2_LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ModelV2] Parameters: {n_params:,}")

    # ── CSV log ───────────────────────────────────────────────────────────
    log_file = open(V2_LOG_PATH, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, V2_NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}"])

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{V2_NUM_EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")

            # Show sample predictions from val set
            model.eval()
            shown_sign, shown_none = False, False
            with torch.no_grad():
                for padded_seqs, targets, input_lengths, target_lengths in val_loader:
                    padded_seqs   = padded_seqs.to(device)
                    input_lengths = input_lengths.to(device)
                    log_probs     = model(padded_seqs)
                    decoded       = ctc_greedy_decode(log_probs.cpu(), input_lengths.cpu())

                    # Walk through the batch to find one sign and one none sample
                    offset = 0
                    for b in range(padded_seqs.shape[1]):
                        tlen = target_lengths[b].item()
                        if tlen > 0:
                            gt = indices_to_glosses(targets[offset:offset + tlen].tolist())
                            offset += tlen
                        else:
                            gt = []
                        pred = indices_to_glosses(decoded[b])

                        if not shown_sign and tlen > 0:
                            print(f"         [SIGN]  GT={gt}  Pred={pred}")
                            shown_sign = True
                        if not shown_none and tlen == 0:
                            print(f"         [NONE]  GT=[]  Pred={pred}")
                            shown_none = True
                        if shown_sign and shown_none:
                            break
                    if shown_sign and shown_none:
                        break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val_loss,
            }, V2_MODEL_PATH)

    log_file.close()
    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")
    print(f"       Model saved  : {V2_MODEL_PATH}")
    print(f"       Training log : {V2_LOG_PATH}")


if __name__ == "__main__":
    main()
