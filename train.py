"""
train.py — Training loop for the LSTM + CTC sign language model.

Usage
-----
  python train.py

Outputs
-------
  model_checkpoint.pt  — best model weights (lowest val loss)
  training_log.csv     — epoch-by-epoch loss history
"""

import os
import csv
import torch
import torch.nn as nn

from config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    BLANK_IDX,
    VOCAB,
)
from dataset import build_dataloaders
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# CTC greedy decode
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs, input_lengths, blank=BLANK_IDX):
    """
    Greedy (best-path) CTC decode.

    Parameters
    ----------
    log_probs     : (T, N, C) — log-softmax output
    input_lengths : (N,)

    Returns
    -------
    list of lists — decoded token indices per batch item
    """
    # Argmax over vocab dimension
    indices = log_probs.argmax(dim=-1)   # (T, N)
    indices = indices.permute(1, 0)      # (N, T)

    decoded_batch = []
    for b in range(indices.shape[0]):
        T   = input_lengths[b].item()
        seq = indices[b, :T].tolist()

        # Collapse repeats then remove blank
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
            padded_seqs    = padded_seqs.to(device)     # (T_max, N, 288)
            targets        = targets.to(device)          # (sum_S,)
            input_lengths  = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(padded_seqs)               # (T_max, N, C)

            # CTCLoss expects input_lengths ≤ T (already true — padded to T_max)
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
    print(f"[Train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────
    model     = build_model(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15,
    )

    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── CSV log ───────────────────────────────────────────────────────────
    log_path = os.path.join(os.path.dirname(MODEL_PATH), "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_loss"])

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{NUM_EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

            # Show one sign prediction and one none prediction from val set
            model.eval()
            shown_sign, shown_none = False, False
            with torch.no_grad():
                for padded_seqs, targets, input_lengths, target_lengths in val_loader:
                    padded_seqs   = padded_seqs.to(device)
                    input_lengths = input_lengths.to(device)
                    log_probs     = model(padded_seqs)
                    decoded       = ctc_greedy_decode(log_probs.cpu(), input_lengths.cpu())

                    for b in range(padded_seqs.shape[1]):
                        tlen = target_lengths[b].item()
                        gt   = indices_to_glosses(targets[:tlen].tolist()) if tlen > 0 else []
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
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   best_val_loss,
            }, MODEL_PATH)

    log_file.close()
    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")
    print(f"       Model saved  : {MODEL_PATH}")
    print(f"       Training log : {log_path}")


if __name__ == "__main__":
    main()
