import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
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
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SignLanguageCNN(nn.Module):
    """
    SE-Enhanced CNN model for sign language recognition.
    Must match the training architecture exactly.
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
        self.fc_input_size = 256 * 14 * 14
        
        # Classification layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Feature extraction with SE blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
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


class SESignLanguagePredictor:
    """
    Complete inference system for SE-enhanced sign language recognition.
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the predictor with a trained SE-enhanced model.
        
        Args:
            model_path: Path to the .pth checkpoint file
            device: torch device (cuda/cpu), auto-detected if None
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"🚀 Using device: {self.device}")
        
        # Initialize SE-enhanced model
        self.model = SignLanguageCNN(num_classes=26, dropout_rate=0.3, se_reduction=16)
        
        # Load checkpoint
        self.load_checkpoint(model_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Class names (A-Z)
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
        # Preprocessing transform (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ SE-Enhanced model loaded successfully!")
        print(f"✓ Ready to predict {len(self.class_names)} sign language letters (A-Z)")
        print(f"✓ Model uses Squeeze-and-Excitation blocks for improved accuracy")
    
    def load_checkpoint(self, model_path):
        """Load model weights from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Print training info if available
            if 'best_val_acc' in checkpoint:
                print(f"📊 Model validation accuracy: {checkpoint['best_val_acc']:.2f}%")
            if 'epoch' in checkpoint:
                print(f"📈 Trained for {checkpoint['epoch']+1} epochs")
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔧 Model parameters: {total_params:,}")
    
    def predict_image(self, image_path, top_k=3):
        """
        Predict sign language letter from an image file.
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        predictions = []
        for i in range(top_k):
            predictions.append({
                'letter': self.class_names[top_indices[i]],
                'confidence': float(top_probs[i]) * 100,
                'probability': float(top_probs[i])
            })
        
        return {
            'top_prediction': predictions[0],
            'all_predictions': predictions,
            'original_image': original_image
        }
    
    def predict_from_array(self, image_array):
        """
        Predict from numpy array (useful for video/webcam).
        
        Args:
            image_array: Numpy array in RGB format
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_letter = self.class_names[predicted_class.item()]
        confidence_score = confidence.item() * 100
        
        return {
            'letter': predicted_letter,
            'confidence': confidence_score
        }
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with confidence scores and SE-block emphasis.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        result = self.predict_image(image_path, top_k=5)
        
        # Create figure with enhanced styling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('SE-Enhanced Sign Language Recognition', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Display image
        ax1.imshow(result['original_image'])
        ax1.axis('off')
        
        # Enhanced prediction display
        pred_text = f"Predicted: {result['top_prediction']['letter']}"
        conf_text = f"Confidence: {result['top_prediction']['confidence']:.1f}%"
        
        # Color based on confidence
        conf_val = result['top_prediction']['confidence']
        if conf_val >= 90:
            color = 'darkgreen'
        elif conf_val >= 70:
            color = 'orange'
        else:
            color = 'red'
        
        ax1.set_title(f"{pred_text}\n{conf_text}", 
                     fontsize=16, fontweight='bold', color=color, pad=20)
        
        # Display top predictions with gradient colors
        letters = [p['letter'] for p in result['all_predictions']]
        confidences = [p['confidence'] for p in result['all_predictions']]
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.7, 0.2, 5))
        
        bars = ax2.barh(letters, confidences, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Confidence (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Top 5 Predictions (SE-CNN)', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            ax2.text(conf + 1.5, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.1f}%', va='center', fontsize=11, fontweight='bold')
        
        # Add SE watermark
        fig.text(0.99, 0.01, 'Powered by SE Blocks', 
                ha='right', va='bottom', fontsize=9, style='italic', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_predict(self, image_folder, output_csv=None, show_progress=True):
        """
        Predict for all images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            output_csv: Optional path to save results as CSV
            show_progress: Show progress bar
            
        Returns:
            List of prediction results
        """
        results = []
        image_files = []
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(image_folder).glob(ext))
        
        print(f"\n{'='*60}")
        print(f"📁 Found {len(image_files)} images in {image_folder}")
        print(f"{'='*60}\n")
        
        correct_predictions = 0
        
        for idx, img_path in enumerate(image_files, 1):
            try:
                result = self.predict_image(str(img_path), top_k=1)
                
                # Extract true label from filename if present (e.g., "A_001.jpg" -> "A")
                filename = img_path.name
                true_label = filename.split(' ')[0].upper() if ' ' in filename else None
                
                is_correct = (true_label == result['top_prediction']['letter']) if true_label else None
                if is_correct:
                    correct_predictions += 1
                
                results.append({
                    'filename': filename,
                    'predicted_letter': result['top_prediction']['letter'],
                    'confidence': result['top_prediction']['confidence'],
                    'true_label': true_label,
                    'correct': is_correct
                })
                
                # Progress indicator
                if show_progress:
                    status = "✓" if is_correct else ("✗" if is_correct is False else "•")
                    print(f"{status} [{idx}/{len(image_files)}] {filename}: "
                          f"{result['top_prediction']['letter']} "
                          f"({result['top_prediction']['confidence']:.1f}%)")
                
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
        
        # Calculate accuracy if true labels available
        labeled_results = [r for r in results if r['true_label'] is not None]
        if labeled_results:
            accuracy = (correct_predictions / len(labeled_results)) * 100
            print(f"\n{'='*60}")
            print(f"📊 Batch Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(labeled_results)})")
            print(f"{'='*60}\n")
        
        # Save to CSV if requested
        if output_csv and results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"💾 Results saved to {output_csv}")
        
        return results
    
    def predict_webcam(self, mirror=True, show_fps=True):
        """
        Real-time prediction from webcam feed with SE-enhanced model.
        
        Args:
            mirror: Whether to mirror the webcam feed
            show_fps: Display FPS counter
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n" + "="*60)
        print("🎥 SE-Enhanced Real-Time Sign Language Recognition")
        print("="*60)
        print("Controls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save current frame")
        print("  • Press 'r' to reset prediction")
        print("="*60 + "\n")
        
        frame_count = 0
        predicted_letter = ""
        confidence = 0.0
        fps = 0
        
        import time
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if mirror:
                frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make prediction every 5 frames (for performance)
            if frame_count % 5 == 0:
                try:
                    result = self.predict_from_array(rgb_frame)
                    predicted_letter = result['letter']
                    confidence = result['confidence']
                except Exception as e:
                    predicted_letter = "Error"
                    confidence = 0.0
            
            # Calculate FPS
            if show_fps:
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
            
            # Create info panel
            h, w = frame.shape[:2]
            panel_height = 120
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)  # Dark gray background
            
            # Draw prediction info
            text = f"Sign: {predicted_letter}"
            conf_text = f"Confidence: {confidence:.1f}%"
            
            # Color based on confidence
            if confidence >= 80:
                color = (0, 255, 0)  # Green
            elif confidence >= 60:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            cv2.putText(panel, text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(panel, conf_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show FPS
            if show_fps:
                cv2.putText(panel, f"FPS: {fps:.1f}", (w-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show SE indicator
            cv2.putText(panel, "SE-CNN", (w-150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            
            # Combine frame and panel
            display = np.vstack([frame, panel])
            
            # Display frame
            cv2.imshow('SE-Enhanced Sign Language Recognition', display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'se_captured_sign_{predicted_letter}_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('r'):
                predicted_letter = ""
                confidence = 0.0
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam session ended")
    
    def compare_with_baseline(self, test_images_folder, baseline_model_path=None):
        """
        Compare SE-enhanced model with baseline CNN.
        
        Args:
            test_images_folder: Path to test images
            baseline_model_path: Optional path to baseline model for comparison
        """
        print("\n" + "="*60)
        print("📊 SE-Enhanced Model Performance Analysis")
        print("="*60 + "\n")
        
        # Predict with SE model
        se_results = self.batch_predict(test_images_folder, show_progress=False)
        
        # Calculate metrics
        if se_results and any(r['true_label'] for r in se_results):
            correct = sum(1 for r in se_results if r['correct'])
            total = sum(1 for r in se_results if r['true_label'] is not None)
            accuracy = (correct / total * 100) if total > 0 else 0
            avg_confidence = np.mean([r['confidence'] for r in se_results])
            
            print(f"SE-Enhanced Model Results:")
            print(f"  • Accuracy: {accuracy:.2f}%")
            print(f"  • Average Confidence: {avg_confidence:.2f}%")
            print(f"  • Total Predictions: {len(se_results)}")
            print(f"  • Correct: {correct}/{total}")
            
            return {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'results': se_results
            }
        else:
            print("⚠️  No labeled test images found (filename format: LABEL_xxx.jpg)")
            return None


def main():
    """
    Example usage of the SE-enhanced inference system.
    """
    print("="*70)
    print(" 🔥 SE-ENHANCED SIGN LANGUAGE RECOGNITION - INFERENCE SYSTEM 🔥")
    print("="*70)
    
    # Path to your trained SE-enhanced model
    MODEL_PATH = 'best_se_sign_language_model.pth'
    
    # Initialize predictor
    try:
        predictor = SESignLanguagePredictor(MODEL_PATH)
    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please make sure your SE-enhanced .pth file is in the current directory")
        print("or update MODEL_PATH with the correct path.")
        return
    
    print("\n" + "="*70)
    print("Choose an option:")
    print("="*70)
    print("1. 🖼️  Predict single image")
    print("2. 📁 Predict batch of images")
    print("3. 🎥 Real-time webcam prediction")
    print("4. 📊 Performance analysis on test set")
    print("5. 🚪 Exit")
    print("="*70)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            result = predictor.predict_image(image_path, top_k=5)
            print(f"\n{'='*70}")
            print(f"🎯 Top Prediction: {result['top_prediction']['letter']}")
            print(f"📈 Confidence: {result['top_prediction']['confidence']:.2f}%")
            print(f"{'='*70}")
            print("\n🏆 Top 5 Predictions:")
            for i, pred in enumerate(result['all_predictions'], 1):
                print(f"  {i}. {pred['letter']}: {pred['confidence']:.2f}%")
            
            # Visualize
            predictor.visualize_prediction(image_path, 
                                          save_path='se_prediction_result.png')
        else:
            print(f"❌ Error: Image file '{image_path}' not found!")
    
    elif choice == '2':
        folder_path = input("Enter folder path: ").strip()
        if os.path.exists(folder_path):
            output_csv = input("Enter output CSV name (or press Enter to skip): ").strip()
            if not output_csv:
                output_csv = None
            predictor.batch_predict(folder_path, output_csv)
        else:
            print(f"❌ Error: Folder '{folder_path}' not found!")
    
    elif choice == '3':
        print("\n🎥 Starting webcam...")
        predictor.predict_webcam(mirror=True, show_fps=True)
    
    elif choice == '4':
        folder_path = input("Enter test images folder path: ").strip()
        if os.path.exists(folder_path):
            predictor.compare_with_baseline(folder_path)
        else:
            print(f"❌ Error: Folder '{folder_path}' not found!")
    
    elif choice == '5':
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
