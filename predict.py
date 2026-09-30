"""CLI prediction plus human-readable SHAP feature contributions."""
import argparse
from explain import explain_record

FEATURES = ["duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins", "connection_rate", "syn_ratio"]


def main():
    parser = argparse.ArgumentParser(description="Explain an IDS prediction for supplied flow features")
    for name in FEATURES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, required=True)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    record = {name: getattr(args, name) for name in FEATURES}
    result = explain_record(record, top_n=args.top)
    print(f"Classification: {result['classification']}")
    print(f"Intrusion probability: {result['intrusion_probability']:.1%}")
    print("\nTop SHAP contributions for this prediction:")
    for item in result["top_contributions"]:
        print(f"  {item['feature']:<18} value={item['value']:<10g} SHAP={item['shap_value']:+.4f} ({item['direction']})")
    print("\nA SHAP contribution explains this model's prediction; it is not proof of causality.")


if __name__ == "__main__":
    main()
