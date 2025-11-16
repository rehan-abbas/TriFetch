import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.data_loader import list_all_episodes, load_episode
from app.preprocess import preprocess_ecg
from app.feature_extraction import compute_basic_features


def train_model():
    episodes = list_all_episodes()

    X = []
    y = []

    for ep in episodes:
        metadata, ecg = load_episode(ep["path"])
        clean = preprocess_ecg(ecg)

        features = compute_basic_features(clean)

        X.append(features)
        y.append(ep["event_type"])   # AF / VTACH / PAUSE

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    clf.fit(X, y)

    # save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/ecg_classifier.pkl")

    print("Model trained and saved to models/ecg_classifier.pkl")


if __name__ == "__main__":
    train_model()
