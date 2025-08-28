import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch3d.ops import knn_points
import os

from model import Image2PCgen
from dataset import get_dataloaders
from utils import EarlyStopping, save_checkpoint, load_checkpoint, cosine_scheduler


# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
ALPHA = 5.0
CHECKPOINT_PATH = "last_checkpoint.pth"
BEST_CHECKPOINT_PATH = "1024_airplane_car.pth"


# --- Loss Function ---
def chamfer_distance_alpha(pred, target, alpha=5.0):
    """
    Computes the asymmetric Chamfer distance.
    The parameter 'alpha' weights the distance from the target to the prediction.
    """
    dist1, _, _ = knn_points(pred, target, K=1)
    dist2, _, _ = knn_points(target, pred, K=1)
    return dist1.mean() + alpha * dist2.mean()


def train_one_epoch(model, optimizer, scheduler, train_loader, epoch, scaler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, pcs in progress_bar:
        images, pcs = images.to(DEVICE), pcs.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            preds = model(images)
        
        preds = preds.float()
        loss = chamfer_distance_alpha(preds, pcs, alpha=ALPHA)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item())

    return running_loss / len(train_loader.dataset)


def validate_one_epoch(model, val_loader, epoch):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for images, pcs in progress_bar:
            images, pcs = images.to(DEVICE), pcs.to(DEVICE)
            
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(images)
            
            preds = preds.float()
            loss = chamfer_distance_alpha(preds, pcs, alpha=ALPHA)
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

    return running_loss / len(val_loader.dataset)


def main():
    train_loader, val_loader, _ = get_dataloaders()

    model = Image2PCgen(num_points=1024).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = cosine_scheduler(optimizer, WARMUP_STEPS, NUM_EPOCHS * len(train_loader))

    start_epoch, train_losses, val_losses, best_val_loss = 1, [], [], float("inf")
    early_stopper = EarlyStopping(patience=5)

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        tqdm.write(f"Resuming from checkpoint {CHECKPOINT_PATH}")
        start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, scaler, DEVICE
        )
        early_stopper.best_loss = best_val_loss
        tqdm.write(f"Resumed at epoch {start_epoch}, last val_loss={val_losses[-1]:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, epoch, scaler)
        val_loss = validate_one_epoch(model, val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        # Always save the last checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler, epoch+1,
                        train_losses, val_losses, early_stopper.best_loss, CHECKPOINT_PATH)

        # Early stopping step
        prev_best = early_stopper.best_loss
        early_stopper.step(val_loss)

        # Save best checkpoint
        if early_stopper.counter == 0 and early_stopper.best_loss < prev_best:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch+1,
                            train_losses, val_losses, early_stopper.best_loss, BEST_CHECKPOINT_PATH)
            tqdm.write(f"New best checkpoint saved at epoch {epoch} with val_loss={val_loss:.4f}")

        if early_stopper.early_stop:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break


import warnings
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()