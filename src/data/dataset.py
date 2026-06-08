import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.model.feature.cnn_1d_4ha import FeatureExtractor
from PIL import Image

class OriginalDataset(Dataset): 
    def __init__(self, df, conf, target_col, y_min, y_max):
        self.df = df.reset_index(drop=True).copy()
        self.conf = conf
        self.target_col = target_col
        self.y_min = np.asarray(y_min, dtype=np.float32)
        self.y_max = np.asarray(y_max, dtype=np.float32)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        texture_path = row["texture_path"]
        texture_image = Image.open(texture_path).convert("L")
        texture_image = self.transform(texture_image)

        if self.conf['dataset_input'] == 'texture_maps':
            height_path = row["height_path"]
            height_map = Image.open(height_path).convert("L")
            height_map = self.transform(height_map)

            normal_path = row["normal_path"]
            normal_map = Image.open(normal_path).convert("L")
            normal_map = self.transform(normal_map)
        
        if self.target_col == "haptic_attribute": # (4,)
            label = np.array(row[self.target_col], dtype=np.float32)
        elif self.target_col == "roughness":
            label = np.array([row[self.target_col]], dtype=np.float32) # (1,)

        label = (label - self.y_min) / (self.y_max - self.y_min + 1e-8)

        if self.conf['dataset_input'] == 'texture_maps':
            return texture_image, height_map, normal_map,  torch.tensor(label, dtype=torch.float32)    
        elif self.conf['dataset_input'] == 'texture_image':
            return texture_image, torch.tensor(label, dtype=torch.float32)
        
def load_original_dataset(df, conf, target_col, y_min, y_max):
    return OriginalDataset(df, conf, target_col, y_min, y_max)

class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
def load_cnn_1d_dataset(df, conf, target_col, y_min, y_max):
    features, targets = FeatureExtractor.precompute_features_and_targets(df, conf, target_col, y_min, y_max)
    return FeatureDataset(features, targets)
