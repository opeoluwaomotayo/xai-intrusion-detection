from generate_dataset import generate
from train import train_model
from explain import explain_record


def test_training_and_explanation(tmp_path):
    data = tmp_path / "traffic.csv"
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    generate(str(data), n=200)
    result = train_model(str(data), str(model), str(metrics))
    assert model.exists() and metrics.exists()
    assert 0 <= result["f1"] <= 1
    explanation = explain_record({
        "duration": 1.0, "src_bytes": 8000, "dst_bytes": 400,
        "packet_count": 300, "failed_logins": 5,
        "connection_rate": 20, "syn_ratio": 0.9,
    }, str(model), top_n=3)
    assert explanation["classification"] in {"NORMAL-LIKE", "INTRUSION-LIKE"}
    assert 0 <= explanation["intrusion_probability"] <= 1
    assert len(explanation["top_contributions"]) == 3
    assert all("shap_value" in item for item in explanation["top_contributions"])
