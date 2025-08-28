import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle

# --- Dataset Configuration ---
IMAGE_PATH = "ShapeNet/ShapeNetRendering"
POINTCLOUD_PATH = "ShapeNet/ShapeNet_pointclouds"
CACHE_FILE = "shapenet_paths.pkl"
CATEGORIES = ['02691156', '02958343'] #Airplane and Car
NUM_POINTS = 1024

# --- DataLoader Configuration ---
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# --- Preprocessing Config ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# --- Dataset Class ---
class ShapeNetImage2PCDataset(Dataset):
    def __init__(self, data_items, transform=None):
        self.items = data_items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.items[idx]
        image_path = item['image_path']
        pc_path = item['pc_path']

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load point cloud
        points = np.load(pc_path).astype(np.float32)

        # Point cloud normalization
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_distance > 0:
            points /= max_distance
        
        points = torch.from_numpy(points)
        
        return image, points


# --- Data Item Collection ---
def get_data_items(image_root, pc_root, categories):
    all_items = []
    for category in categories:
        pc_category_path = os.path.join(pc_root, category)
        
        # Check if the point cloud category path exists before proceeding
        if not os.path.isdir(pc_category_path):
            continue

        for model_id in os.listdir(pc_category_path):
            image_model_path = os.path.join(image_root, category, model_id, 'rendering')
            
            if not os.path.isdir(image_model_path):
                continue

            pc_path = os.path.join(pc_category_path, model_id, f'pointcloud_{NUM_POINTS}.npy')
            if not os.path.exists(pc_path):
                continue
            
            image_paths = sorted(glob.glob(os.path.join(image_model_path, '*.png')))
            
            for image_path in image_paths:
                all_items.append({
                    'image_path': image_path,
                    'pc_path': pc_path,
                    'category': category,
                    'model_id': model_id
                })
                
    print(f"Path pre-computation finished. Found {len(all_items)} samples.")
    return all_items


# --- Main Data Loader Function ---
def get_dataloaders():
    if os.path.exists(CACHE_FILE):
        print("Loading dataset file paths from cache...")
        with open(CACHE_FILE, 'rb') as f:
            all_items = pickle.load(f)
        print(f"Loaded {len(all_items)} samples from cache.")
    else:
        print("Cache file not found. Pre-computing dataset file paths. This may take a while...")
        all_items = get_data_items(IMAGE_PATH, POINTCLOUD_PATH, CATEGORIES)
        
        if len(all_items) == 0:
            raise ValueError("Dataset is empty. Please check your data paths and categories.")
            
        print(f"Saving file paths to cache at {CACHE_FILE}...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_items, f)
        print("File paths saved. Subsequent runs will be much faster.")

    dataset = ShapeNetImage2PCDataset(
        data_items=all_items,
        transform=transform
    )

    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size

    # Set a fixed seed for reproducibility before splitting the dataset
    torch.manual_seed(RANDOM_SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )

    print(f"Total number of samples: {total_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader