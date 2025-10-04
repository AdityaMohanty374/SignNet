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


class SignLanguageCNN(nn.Module):
    """
    CNN model for sign language recognition (must match training architecture).
    """
    def __init__(self, num_classes=26, input_channels=3, dropout_rate=0.2):
        super(SignLanguageCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
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
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
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


class SignLanguagePredictor:
    """
    Complete inference system for sign language recognition.
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the .pth checkpoint file
            device: torch device (cuda/cpu), auto-detected if None
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SignLanguageCNN(num_classes=26, dropout_rate=0.2)
        
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
        
        print("Model loaded successfully!")
        print(f"Ready to predict {len(self.class_names)} sign language letters (A-Z)")
    
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
                print(f"Model validation accuracy: {checkpoint['best_val_acc']:.2f}%")
            if 'epoch' in checkpoint:
                print(f"Trained for {checkpoint['epoch']+1} epochs")
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
    
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
        Visualize prediction with confidence scores.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        result = self.predict_image(image_path, top_k=5)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(result['original_image'])
        ax1.axis('off')
        ax1.set_title(f"Predicted: {result['top_prediction']['letter']} "
                     f"({result['top_prediction']['confidence']:.1f}%)",
                     fontsize=16, fontweight='bold', color='green')
        
        # Display top predictions
        letters = [p['letter'] for p in result['all_predictions']]
        confidences = [p['confidence'] for p in result['all_predictions']]
        
        colors = ['green', 'blue', 'orange', 'red', 'purple']
        bars = ax2.barh(letters, confidences, color=colors)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            ax2.text(conf + 1, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_predict(self, image_folder, output_csv=None):
        """
        Predict for all images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            output_csv: Optional path to save results as CSV
            
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
    
    def predict_webcam(self, mirror=True):
        """
        Real-time prediction from webcam feed.
        
        Args:
            mirror: Whether to mirror the webcam feed
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        frame_count = 0
        
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
            
            # Display prediction on frame
            text = f"Sign: {predicted_letter} ({confidence:.1f}%)"
            cv2.putText(frame, text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Display frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'captured_sign_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Example usage of the inference system.
    """
    print("=" * 60)
    print("Sign Language Recognition - Inference System")
    print("=" * 60)
    
    # Path to your trained model
    MODEL_PATH = 'best_sign_language_model.pth'
    
    # Initialize predictor
    try:
        predictor = SignLanguagePredictor(MODEL_PATH)
    except FileNotFoundError:
        print(f"\nError: Model file '{MODEL_PATH}' not found!")
        print("Please make sure your .pth file is in the current directory")
        print("or update MODEL_PATH with the correct path.")
        return
    
    print("\n" + "=" * 60)
    print("Choose an option:")
    print("=" * 60)
    print("1. Predict single image")
    print("2. Predict batch of images")
    print("3. Real-time webcam prediction")
    print("4. Exit")
    print("=" * 60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            result = predictor.predict_image(image_path, top_k=5)
            print(f"\n{'='*60}")
            print(f"Top Prediction: {result['top_prediction']['letter']}")
            print(f"Confidence: {result['top_prediction']['confidence']:.2f}%")
            print(f"{'='*60}")
            print("\nTop 5 Predictions:")
            for i, pred in enumerate(result['all_predictions'], 1):
                print(f"{i}. {pred['letter']}: {pred['confidence']:.2f}%")
            
            # Visualize
            predictor.visualize_prediction(image_path, 
                                          save_path='prediction_result.png')
        else:
            print(f"Error: Image file '{image_path}' not found!")
    
    elif choice == '2':
        folder_path = input("Enter folder path: ").strip()
        if os.path.exists(folder_path):
            output_csv = input("Enter output CSV name (or press Enter to skip): ").strip()
            if not output_csv:
                output_csv = None
            predictor.batch_predict(folder_path, output_csv)
        else:
            print(f"Error: Folder '{folder_path}' not found!")
    
    elif choice == '3':
        print("\nStarting webcam...")
        predictor.predict_webcam(mirror=True)
    
    elif choice == '4':
        print("Goodbye!")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
