import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import yaml
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import argparse


class UlcerStagingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_stage_classifier(config_path="configs/stage_config.yaml", train_dir="dataset_staged/train", val_dir="dataset_staged/val"):
    """
    Train the ulcer staging classifier model with integrity checks and class balancing
    """
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Canonical config not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    canonical_classes = config['stage_classifier']['canonical_order']
    
    # Pre-training Audit: Enforce class presence and non-empty status (Comment 1)
    print("Performing dataset integrity audit...")
    for target_dir, label in [(train_dir, "Training"), (val_dir, "Validation")]:
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"{label} directory missing: {target_dir}")
        
        present_dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        for cls in canonical_classes:
            if cls not in present_dirs:
                raise ValueError(f"CRITICAL: {label} class missing! Expected '{cls}' in {target_dir}")
            
            img_count = len([f for f in os.listdir(os.path.join(target_dir, cls)) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if img_count == 0:
                raise ValueError(f"CRITICAL: {label} class '{cls}' is empty in {target_dir}")
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((config['stage_classifier']['input_size'], config['stage_classifier']['input_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['stage_classifier']['input_size'], config['stage_classifier']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using canonical class mapping
    class_to_idx = {cls: i for i, cls in enumerate(canonical_classes)}
    print(f"Using class mapping: {class_to_idx}")
    
    train_dataset = UlcerStagingDataset(train_dir, transform=train_transform)
    train_dataset.class_to_idx = class_to_idx
    train_dataset.samples = []
    for cls in canonical_classes:
        class_dir = os.path.join(train_dir, cls)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_dataset.samples.append((os.path.join(class_dir, img_name), class_to_idx[cls]))

    val_dataset = UlcerStagingDataset(val_dir, transform=val_transform)
    val_dataset.class_to_idx = class_to_idx
    val_dataset.samples = []
    for cls in canonical_classes:
        class_dir = os.path.join(val_dir, cls)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                val_dataset.samples.append((os.path.join(class_dir, img_name), class_to_idx[cls]))
    
    # Compute Class Weights for Imbalance Handling (Comment 3)
    labels = [s[1] for s in train_dataset.samples]
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    full_weights = torch.ones(len(canonical_classes))
    for i, label in enumerate(unique_labels):
        full_weights[label] = class_weights[i]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_weights = torch.tensor(full_weights, dtype=torch.float32).to(device)
    print(f"Applying class weights: {full_weights}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Load pre-trained model
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config['stage_classifier']['num_classes'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=full_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct_val / total_val
        avg_val_loss = val_running_loss / len(val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save Mapping and Order for Inference (Comment 2)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'canonical_order': canonical_classes,
                'val_acc': val_acc,
            }, config['stage_classifier']['model_path'])
            print(f"New best model saved! Accuracy: {best_val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
    
    print(f'Finished Training. Best accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ulcer staging classifier')
    parser.add_argument('--config', type=str, default='configs/stage_config.yaml', help='Path to config file')
    parser.add_argument('--train_dir', type=str, default='dataset_staged/train', help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='dataset_staged/val', help='Path to validation data directory')
    args = parser.parse_args()
    train_stage_classifier(args.config, args.train_dir, args.val_dir)