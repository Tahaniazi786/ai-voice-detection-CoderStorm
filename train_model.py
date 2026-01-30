import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# Dummy but LEGIT dataset (temporary)
# -----------------------------------
# Later you can replace this with real data

# 200 samples, 13 MFCC features
X = np.random.rand(200, 13)

# Labels: 0 = HUMAN, 1 = AI_GENERATED
y = np.array([0]*100 + [1]*100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test accuracy (just for sanity)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
