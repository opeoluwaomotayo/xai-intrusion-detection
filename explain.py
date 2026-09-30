"""Local SHAP explanations for authorized/synthetic network-flow records."""
import joblib
import numpy as np
import pandas as pd
import shap


def _positive_class_values(explanation):
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base.ndim == 2:
            base = base[:, 1]
    return values, base


def explain_record(record, model_path="models/xai_ids.joblib", top_n=5):
    bundle = joblib.load(model_path)
    features = bundle["features"]
    frame = pd.DataFrame([[record[name] for name in features]], columns=features)
    model = bundle["model"]
    probability = float(model.predict_proba(frame)[0][1])
    prediction = int(model.predict(frame)[0])

    explainer = shap.TreeExplainer(model)
    explanation = explainer(frame)
    values, base = _positive_class_values(explanation)
    contributions = values[0]
    ranked = sorted(
        [
            {"feature": name, "value": float(frame.iloc[0][name]), "shap_value": float(contribution),
             "direction": "toward intrusion" if contribution > 0 else "toward normal"}
            for name, contribution in zip(features, contributions)
        ],
        key=lambda item: abs(item["shap_value"]), reverse=True,
    )[:top_n]
    base_value = float(np.ravel(base)[0]) if np.size(base) else 0.0
    return {
        "classification": "INTRUSION-LIKE" if prediction else "NORMAL-LIKE",
        "intrusion_probability": probability,
        "base_value": base_value,
        "top_contributions": ranked,
    }
