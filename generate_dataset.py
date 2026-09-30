"""Generate a reproducible synthetic flow dataset for defensive XAI/IDS teaching."""
import csv
import random
from pathlib import Path

FIELDS = ["duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins", "connection_rate", "syn_ratio", "label"]


def generate(path="data/network_traffic.csv", n=600, seed=42):
    rng = random.Random(seed)
    rows = []
    for _ in range(n // 2):
        rows.append([
            round(rng.uniform(0.1, 15), 3), rng.randint(100, 7000), rng.randint(100, 10000),
            rng.randint(5, 120), rng.choice([0, 0, 0, 1]), round(rng.uniform(0.1, 8), 3),
            round(rng.uniform(0.05, 0.55), 3), 0,
        ])
    for _ in range(n - n // 2):
        rows.append([
            round(rng.uniform(0.01, 5), 3), rng.randint(10, 12000), rng.randint(0, 2500),
            rng.randint(80, 600), rng.randint(1, 8), round(rng.uniform(6, 35), 3),
            round(rng.uniform(0.55, 1.0), 3), 1,
        ])
    rng.shuffle(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(f"Generated {generate()} synthetic rows.")
