# Explainable AI Intrusion Detection System (XAI IDS)

A defensive **Cybersecurity + Machine Learning + Explainable AI** teaching project. It trains a Random Forest classifier on reproducible synthetic network-flow data and uses **SHAP** to explain how individual features influence a prediction.

## Why XAI?

A conventional classifier can output an intrusion probability. This project adds a local explanation showing which flow features pushed the model prediction toward `INTRUSION-LIKE` or `NORMAL-LIKE`. These explanations describe model behavior and do **not** establish causality.

## Pipeline

```text
Synthetic Flow Dataset
        ↓
Train/Test Split
        ↓
Random Forest IDS
        ↓
Intrusion Prediction
        ↓
SHAP Tree Explanation
        ↓
Ranked Feature Contributions
```

## Project Structure

```text
xai-intrusion-detection/
├── .github/workflows/python-tests.yml
├── data/network_traffic.csv
├── models/.gitkeep
├── tests/test_xai_ids.py
├── generate_dataset.py
├── train.py
├── explain.py
├── predict.py
├── pytest.ini
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Generate Teaching Data

```bash
python generate_dataset.py
```

## Train

```bash
python train.py
```

## Explain a Prediction

```bash
python predict.py --duration 1 --src-bytes 8000 --dst-bytes 400 --packet-count 300 --failed-logins 5 --connection-rate 20 --syn-ratio 0.9
```

The program prints the prediction, its model probability, and the highest-magnitude SHAP contributions.

## Tests and GitHub Actions

```bash
python -m pytest -v
```

The bundled GitHub Actions workflow uses the same `python -m pytest -v` command.

## Security Scope

This repository processes supplied numeric flow features only. It does not sniff traffic, scan systems, send packets, exploit services, or connect to targets. Use only authorized data and environments.

## Scientific Limitations

- The bundled data is synthetic and designed for teaching/CI.
- Its performance must not be presented as real-world IDS accuracy.
- SHAP values explain the fitted model, not the true causal mechanism behind an attack.
- A research version needs an appropriately licensed real IDS dataset, leakage controls, class-imbalance analysis, robust validation, and comparison with baseline/explanation methods.

## Suggested Research Extensions

1. Replace synthetic data with a documented benchmark dataset.
2. Compare Random Forest, XGBoost/Gradient Boosting, and neural approaches.
3. Add global SHAP feature importance and local waterfall plots.
4. Compare explanation stability across models and samples.
5. Measure both predictive performance and explanation quality.
6. Build a Streamlit dashboard for analyst-facing explanations.

## Author

**Dr. Opeoluwa Omotayo Ajilore**  
Lecturer, Computer Science

## License

MIT License. See `LICENSE`.
