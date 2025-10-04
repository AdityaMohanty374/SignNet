from SE_CBAM_inference import SECBAMSignLanguagePredictor

# Initialize
predictor = SECBAMSignLanguagePredictor(f"Model Path to {best_se_cbam_model.pth}")

#result = predictor.predict_image(f"dir to a single image")
#print(f"Predicted: {result['top_prediction']['letter']}")

# Visualize results
#predictor.visualize_prediction(f"dir to a single image")


predictor.batch_predict(f"Image Dataset Path", output_csv=f'Save dir for{baseAndSE+CBAM_results.csv}')
