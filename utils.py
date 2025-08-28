import torch
import os
import math
from torch.optim.lr_scheduler import LambdaLR
import clip
import pickle
from PIL import Image
from tqdm import tqdm

from dataset import get_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FILE = "shapenet_paths.pkl"
EMBEDDING_FILE = "image_embeddings.pkl"

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_losses, val_losses, best_val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return (checkpoint['epoch'], 
            checkpoint['train_losses'], 
            checkpoint['val_losses'], 
            checkpoint.get('best_val_loss', float("inf")))


def cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6, base_lr=1e-4):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def precompute_image_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        print(f"Embeddings already exist at {EMBEDDING_FILE}. Skipping pre-computation.")
        return

    print("Loading dataset paths...")
    if not os.path.exists(CACHE_FILE):
        get_dataloaders()
        
    with open(CACHE_FILE, 'rb') as f:
        all_items = pickle.load(f)

    print("Loading CLIP model...")
    model, preprocess = clip.load("RN50", device=DEVICE)
    model.eval()

    image_embeddings = {}
    
    # Process images in batches for efficiency
    BATCH_SIZE = 64
    all_paths = [item['image_path'] for item in all_items]

    for i in tqdm(range(0, len(all_paths), BATCH_SIZE), desc="Computing Embeddings"):
        batch_paths = all_paths[i:i + BATCH_SIZE]
        
        # Load and preprocess batch of images
        images = [preprocess(Image.open(path).convert('RGB')) for path in batch_paths]
        image_batch = torch.stack(images).to(DEVICE)
        
        with torch.no_grad():
            features = model.encode_image(image_batch)
            features /= features.norm(dim=-1, keepdim=True)
        
        # Store results
        for path, feature in zip(batch_paths, features):
            image_embeddings[path] = feature.cpu() # Store on CPU to save GPU memory

    print(f"Saving {len(image_embeddings)} embeddings to {EMBEDDING_FILE}...")
    with open(EMBEDDING_FILE, 'wb') as f:
        pickle.dump(image_embeddings, f)

    print("Pre-computation complete.")
