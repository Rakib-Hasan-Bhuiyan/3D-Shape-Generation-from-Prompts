import os
import json
import torch
from tqdm import tqdm
import kaolin.metrics.pointcloud as pc_metrics
import clip
import pickle
from PIL import Image

from dataset import transform
from dataset import get_dataloaders
from model import Image2PCgen

EMBEDDING_FILE = "image_embeddings.pkl"
CHECKPOINT_PATH = "1024_aiplane_car.pth"

CATEGORY_NAMES = {
    '02691156': 'Airplane',
    '02958343': 'Car'
}

def evaluate_model(model, test_loader, device, model_name, checkpoint_path, results_file="results.json"):
    
    # --- ADDED CHECKPOINT LOADING ---
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Evaluating with current model state.")
    # --- END OF ADDITION ---
    
    model.eval()
    category_metrics = {cat: {"cd": [], "fscore": []} for cat in CATEGORY_NAMES.keys()}

    with torch.no_grad():
        for batch_idx, (images, pcs) in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            pcs = pcs.to(device)

            # Handle PointCloudNet (needs [B, num_views, C, H, W])
            if model.__class__.__name__ == "PointCloudNet":
                images = images.unsqueeze(1)

            images = images.to(device)
            preds = model(images).float()  # (B, N, 3)

            # Metrics per sample
            cd_vals = pc_metrics.chamfer_distance(preds, pcs)  # (B,)
            f_vals = pc_metrics.f_score(pcs, preds, radius=0.01)  # (B,)

            # Map results to categories for this batch
            start_idx = batch_idx * test_loader.batch_size
            for i in range(len(images)):
                global_idx = test_loader.dataset.indices[start_idx + i]
                cat_id = test_loader.dataset.dataset.items[global_idx]['category']
                if cat_id in category_metrics:
                    category_metrics[cat_id]['cd'].append(cd_vals[i].item())
                    category_metrics[cat_id]['fscore'].append(f_vals[i].item())

    # --- Average per category ---
    results = {}
    for cat_id, metrics in category_metrics.items():
        if len(metrics['cd']) == 0:
            continue
        results[CATEGORY_NAMES[cat_id]] = {
            "Chamfer": sum(metrics['cd']) / len(metrics['cd']),
            "F-score": sum(metrics['fscore']) / len(metrics['fscore']),
        }

    # --- Save results ---
    all_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    all_results[model_name] = results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved {model_name} results to {results_file}")
    return results


def retrieve_best_image(prompt: str, device: torch.device):

    if not os.path.exists(EMBEDDING_FILE):
        print(f"Error: Embeddings file not found at {EMBEDDING_FILE}.")
        return None
    
    # Load Pre-computed Image Embeddings
    with open(EMBEDDING_FILE, 'rb') as f:
        image_embeddings_dict = pickle.load(f)
    
    image_paths = list(image_embeddings_dict.keys())
    image_features = torch.stack(list(image_embeddings_dict.values())).to(device)

    # Load CLIP Model (only the text encoder)
    model, _ = clip.load("RN50", device=device)
    model.eval()

    # Encode Text Prompt
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize

    # Calculate Cosine Similarity
    # (Text Feature: 1 x D) @ (Image Features: N x D).T -> 1 x N
    similarities = (text_features @ image_features.T).squeeze(0)
    
    # Find the Best Match
    best_idx = similarities.argmax().item()
    best_image_path = image_paths[best_idx]
    best_similarity = similarities[best_idx].item()
    
    return best_image_path


def text_inference(model, device, prompt: str):

    model.eval()
    best_image_path = retrieve_best_image(prompt, device)
    image = Image.open(best_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        preds = model(image_tensor).float()  # (1, N, 3)
    
    return preds



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Image2PCgen().to(device)
    _, _, test_loader = get_dataloaders()
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name="Image2PCgen",
        checkpoint_path=CHECKPOINT_PATH,
    )