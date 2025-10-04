# Simple usage
from inference_system import SignLanguagePredictor

# Initialize with your .pth file
predictor = SignLanguagePredictor("C:\\Users\\mohan\\Desktop\\SignLanguageInterpreter\\BaseModel\\BaseModel.pth")

# Predict single image
result = predictor.predict_image("C:\\Users\\mohan\\Downloads\\image.jpg")
print(f"Predicted: {result['top_prediction']['letter']}")

# Visualize results
predictor.visualize_prediction("C:\\Users\\mohan\\Downloads\\image.jpg")

# Batch process
#predictor.batch_predict('C:\\Users\\mohan\\Desktop\\Test\\Gaussian\\BG_clutter', output_csv='C:\\Users\\mohan\\Desktop\\Test\\Gaussian\\BG_clutter\\base_results.csv')

# Real-time webcam
#predictor.predict_webcam()