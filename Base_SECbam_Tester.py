from SE_CBAM_inference import SECBAMSignLanguagePredictor

# Initialize
predictor = SECBAMSignLanguagePredictor(r"C:\Users\mohan\Desktop\SignLanguageInterpreter\BaseModel+SE+CBAM\best_se_cbam_model.pth")

#result = predictor.predict_image("C:\\Users\\mohan\\Downloads\\image.jpg")
#print(f"Predicted: {result['top_prediction']['letter']}")

# Visualize results
#predictor.visualize_prediction("C:\\Users\\mohan\\Downloads\\image.jpg")

predictor.batch_predict(r'C:\Users\mohan\Desktop\SignLanguageInterpreter\Test\Gaussian', output_csv=r'C:\Users\mohan\Desktop\SignLanguageInterpreter\Test\Gaussian\baseAndSE+CBAM_results.csv')