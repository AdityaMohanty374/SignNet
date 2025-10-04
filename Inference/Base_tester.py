# Simple usage
from Base_inference import SignLanguagePredictor

# Initialize with your .pth file
predictor = SignLanguagePredictor(f"Model path to {BaseModel.pth}")

# Predict single image
result = predictor.predict_image("Single Image Dir")
print(f"Predicted: {result['top_prediction']['letter']}")

# Visualize results
predictor.visualize_prediction("Single Image Dir")

# Batch process
#predictor.batch_predict('Image Dataset Path', output_csv=f'Save Location for: {base_results.csv}')

# Real-time webcam

#predictor.predict_webcam()

