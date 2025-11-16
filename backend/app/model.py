import joblib
from app.feature_extraction import compute_basic_features
from app.preprocess import preprocess_ecg

model = joblib.load("models/ecg_classifier.pkl")

def predict_event_type(ecg_signal):
    clean = preprocess_ecg(ecg_signal)
    features = compute_basic_features(clean)
    label = model.predict([features])[0]
    return label

