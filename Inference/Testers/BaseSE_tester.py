from SE_inference import SESignLanguagePredictor

# Initialize
predictor = SESignLanguagePredictor(f"Model Path to {BaseModelwithSE.pth}")

result = predictor.predict_image(f"Single Image dir")
print(f"Predicted: {result['top_prediction']['letter']}")

# Visualize results
predictor.visualize_prediction(f"Single Image dir")


#predictor.batch_predict(f"Image Dataset path", output_csv=f'Save Location for: {baseAndSE_results.csv}')
