from SE_inference import SESignLanguagePredictor

# Initialize
predictor = SESignLanguagePredictor("C:\\Users\\mohan\\Desktop\\SignLanguageInterpreter\\BaseModel+SE\\BaseModelwithSE.pth")

result = predictor.predict_image("C:\\Users\\mohan\\Downloads\\image.jpg")
print(f"Predicted: {result['top_prediction']['letter']}")

# Visualize results
predictor.visualize_prediction("C:\\Users\\mohan\\Downloads\\image.jpg")

#predictor.batch_predict('C:\\Users\\mohan\\Desktop\\Test\\Gaussian\\BG_clutter', output_csv='C:\\Users\\mohan\\Desktop\\Test\\Gaussian\\BG_clutter\\baseAndSE_results.csv')