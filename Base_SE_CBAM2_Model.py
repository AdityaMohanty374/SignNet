import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from PIL import Image
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ==============================
# Squeeze-and-Excitation Block
# ==============================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==============================
# CBAM (Channel + Spatial Attention)
# ==============================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, stride=1,
                                      padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # ---- Channel Attention ----
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)
        
        # ---- Spatial Attention ----
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att.expand_as(x)
        
        return x


# ==============================
# Hybrid CNN with SE + CBAM
# ==============================
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=26, input_channels=3, dropout_rate=0.3, se_reduction=16):
        super(SignLanguageCNN, self).__init__()
        
        # Block 1 (SE)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32, reduction=se_reduction)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2 (SE)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64, reduction=se_reduction)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3 (SE)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128, reduction=se_reduction)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4 (CBAM only)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.cbam4 = CBAM(256, reduction=se_reduction)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Flatten size (224x224 input → 14x14 after 4 pools)
        self.fc_input_size = 256 * 14 * 14
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)   # Only CBAM here
        x = self.pool4(x)
        
        # Flatten + Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        x = self.fc3(x)
        return x



# ==============================
# Dataset Class
# ==============================
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
    
    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, filename)
                        samples.append((path, self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            with open(path, 'rb') as f:
                image = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


# ==============================
# Data Transforms
# ==============================
def get_data_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms

# ==============================
# Training Manager
# ==============================
class SignLanguageTrainer:
    def __init__(self, model, device, num_classes=26):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.train_losses, self.train_accuracies = [], []
        self.val_losses, self.val_accuracies = [], []

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(dataloader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'Loss': f'{running_loss/(total/targets.size(0)):.4f}', 
                              'Acc': f'{100.*correct/total:.2f}%'})
        return running_loss / len(dataloader), 100.*correct/total

    def validate_epoch(self, dataloader, criterion):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_predictions, all_targets = [], []
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                pbar.set_postfix({'Loss': f'{running_loss/(total/targets.size(0)):.4f}', 
                                  'Acc': f'{100.*correct/total:.2f}%'})
        return running_loss / len(dataloader), 100.*correct/total, all_predictions, all_targets

    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=5e-4):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_val_acc, patience_counter = 0.0, 0
        early_stopping_patience = 15

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}\n" + "-"*50)
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            scheduler.step()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc, patience_counter = val_acc, 0
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_acc': best_val_acc}, 
                           'best_se_cbam_model.pth')
                print(f"✓ New best model saved! Validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping")
                    break
        return val_preds, val_targets

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.legend(); ax1.set_title("Loss")
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.legend(); ax2.set_title("Accuracy")
        plt.show()


# ==============================
# Confusion Matrix
# ==============================
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix (SE+CBAM)'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()


# ==============================
# Inference Class
# ==============================
class SignLanguageInference:
    def __init__(self, model_path, device):
        self.device = device
        self.model = SignLanguageCNN(num_classes=26, dropout_rate=0.3)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device).eval()
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        return self.class_names[pred.item()], conf.item()


# ==============================
# Main Function
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS = 26, 32, 5
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, SE_REDUCTION = 0.001, 5e-4, 0.3, 16

    model = SignLanguageCNN(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, se_reduction=SE_REDUCTION)
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")

    train_tf, val_tf = get_data_transforms()
    data_path = "/content/drive/MyDrive/Train_Data_Sign_Language"

    if os.path.exists(data_path):
        dataset = SignLanguageDataset(data_path, transform=train_tf)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        val_ds.dataset.transform = val_tf
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        trainer = SignLanguageTrainer(model, device, NUM_CLASSES)
        val_preds, val_targets = trainer.train(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
        trainer.plot_training_history()
        class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        plot_confusion_matrix(val_targets, val_preds, class_names)
        print(classification_report(val_targets, val_preds, target_names=class_names))
    else:
        print("Dataset not found at", data_path)


if __name__ == "__main__":
    main()