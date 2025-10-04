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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Paper: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    
    This block adaptively recalibrates channel-wise feature responses by:
    1. Squeezing: Global average pooling to get channel-wise statistics
    2. Excitation: Two FC layers to capture channel dependencies
    3. Scaling: Multiply features by learned channel weights
    """
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
        # Squeeze: Global information embedding
        y = self.squeeze(x).view(b, c)
        # Excitation: Adaptive recalibration
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)


class SignLanguageCNN(nn.Module):
    """
    Enhanced CNN model with SE blocks for improved sign language recognition.
    SE blocks improve generalization by focusing on informative features and
    suppressing less useful ones, reducing overfitting.
    """
    def __init__(self, num_classes=26, input_channels=3, dropout_rate=0.3, se_reduction=16):
        super(SignLanguageCNN, self).__init__()
        
        # Feature extraction layers with SE blocks
        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32, reduction=se_reduction)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64, reduction=se_reduction)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128, reduction=se_reduction)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.se4 = SEBlock(256, reduction=se_reduction)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Calculate the size of flattened features
        # For input size 224x224: after 4 pooling layers -> 14x14
        self.fc_input_size = 256 * 14 * 14
        
        # Classification layers with increased dropout for better regularization
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Feature extraction with SE blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)  # Apply SE block
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)  # Apply SE block
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)  # Apply SE block
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)  # Apply SE block
        x = self.pool4(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class SignLanguageDataset(Dataset):
    """
    Custom dataset class for sign language images with preprocessing.
    """
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


def get_data_transforms():
    """
    Enhanced data augmentation with MixUp-style augmentation capability.
    """
    # Training transforms with stronger augmentation to reduce overfitting
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
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Random erasing for robustness
    ])
    
    # Validation/Test transforms without augmentation
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


class SignLanguageTrainer:
    """
    Enhanced training class with label smoothing for better generalization.
    """
    def __init__(self, model, device, num_classes=26):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, weight_decay=5e-4):
        # Use label smoothing to reduce overfitting
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # AdamW optimizer with better weight decay handling
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 25
        
        print("Starting training with SE-enhanced CNN...")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }, 'best_se_sign_language_model.pth')
                print(f"✓ New best model saved! Validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
        
        return val_preds, val_targets
    
    def plot_training_history(self):
        """Plot training and validation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)
        ax1.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Training and Validation Loss (with SE Blocks)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_title('Training and Validation Accuracy (with SE Blocks)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('se_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix (SE-CNN)'):
    """Plot confusion matrix with proper formatting."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('se_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_sample_dataset_structure():
    """
    Create a sample dataset structure for demonstration.
    """
    print("Sample dataset structure:")
    print("data/")
    print("├── A/")
    print("│   ├── image1.jpg")
    print("│   ├── image2.jpg")
    print("│   └── ...")
    print("├── B/")
    print("│   ├── image1.jpg")
    print("│   └── ...")
    print("├── C/")
    print("└── ... (for all 26 letters)")
    print("\nPlace your sign language images in folders named after each letter A-Z")


def main():
    """
    Main function to run the SE-enhanced sign language recognition system.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced hyperparameters for better generalization
    NUM_CLASSES = 26
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4  # Increased for better regularization
    DROPOUT_RATE = 0.3   # Increased dropout
    SE_REDUCTION = 16    # SE block reduction ratio
    
    # Create SE-enhanced model
    model = SignLanguageCNN(
        num_classes=NUM_CLASSES, 
        dropout_rate=DROPOUT_RATE,
        se_reduction=SE_REDUCTION
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"SE blocks integrated for improved feature recalibration")
    
    # Get transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Dataset path
    data_path = '/content/drive/MyDrive/Train_Data_Sign_Language'
    
    if os.path.exists(data_path):
        # Load dataset
        full_dataset = SignLanguageDataset(data_path, transform=train_transforms)
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Apply different transforms to validation set
        val_dataset.dataset.transform = val_transforms
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create trainer
        trainer = SignLanguageTrainer(model, device, NUM_CLASSES)
        
        # Train model
        val_preds, val_targets = trainer.train(
            train_loader, val_loader, 
            num_epochs=NUM_EPOCHS, 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Create confusion matrix
        class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        plot_confusion_matrix(val_targets, val_preds, class_names)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(val_targets, val_preds, target_names=class_names))
        
    else:
        print(f"Dataset path '{data_path}' not found.")
        create_sample_dataset_structure()
        print("\nTo use this implementation:")
        print("1. Organize your sign language images in the structure shown above")
        print("2. Update the 'data_path' variable with your dataset location")
        print("3. Run the script again")


class SignLanguageInference:
    """
    Enhanced inference class for SE-CNN model.
    """
    def __init__(self, model_path, device):
        self.device = device
        self.model = SignLanguageCNN(num_classes=26, dropout_rate=0.3)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Class names
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict the sign language letter from an image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        predicted_letter = self.class_names[predicted_class.item()]
        confidence_score = confidence.item()
        
        return predicted_letter, confidence_score


if __name__ == "__main__":
    main()