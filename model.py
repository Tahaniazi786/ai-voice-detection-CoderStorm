import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

def predict(features):
    """
    Takes MFCC features and returns:
    label (AI_GENERATED / HUMAN) and confidence score
    """
    features = np.array(features).reshape(1, -1)

    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    predicted_class = model.predict(features)[0]

    if predicted_class == 1:
        label = "AI_GENERATED"
    else:
        label = "HUMAN"

    return label, round(confidence, 2)
