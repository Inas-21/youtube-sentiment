"""Tests d'intégration et benchmark de l'API (locale et/ou déployée).

Pour chaque URL : /health, gestion d'erreurs, temps de réponse selon la taille du batch.
Si deux URLs sont fournies : vérifie que les prédictions sont identiques.

Usage :
    uv run python scripts/integration_test.py --local http://localhost:8000 \
        --remote https://<nom>-7860.app.github.dev
Sortie : reports/integration_report.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
REPORT_PATH = ROOT / "reports" / "integration_report.json"
BATCH_SIZES = [1, 10, 50, 100, 500]
RUNS = 5
TIMEOUT = 60

INVALID_PAYLOADS = {
    "batch vide": {"comments": []},
    "commentaire vide": {"comments": ["ok", "  "]},
    "batch > 500": {"comments": ["x"] * 501},
    "mauvais type": {"comments": "texte"},
    "champ manquant": {"texts": ["a"]},
}


def sample_comments(n: int) -> list[str]:
    texts = pd.read_csv(TEST_PATH)["text"].dropna().tolist()
    return (texts * (n // len(texts) + 1))[:n]


def post(url: str, payload: dict) -> tuple[requests.Response, float]:
    start = time.perf_counter()
    r = requests.post(f"{url}/predict_batch", json=payload, timeout=TIMEOUT)
    return r, (time.perf_counter() - start) * 1000


def check_api(name: str, url: str) -> dict:
    print(f"\n=== {name} : {url} ===")
    result = {"url": url}

    start = time.perf_counter()
    health = requests.get(f"{url}/health", timeout=TIMEOUT)
    result["health"] = {"status_code": health.status_code, "body": health.json(),
                        "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    print(f"/health -> {health.status_code} {health.json().get('status')} "
          f"({result['health']['latency_ms']} ms)")

    result["errors"] = {}
    for label, payload in INVALID_PAYLOADS.items():
        r, _ = post(url, payload)
        result["errors"][label] = r.status_code
        print(f"  {'OK' if r.status_code == 422 else 'KO'} {label:<18} -> HTTP {r.status_code}")

    post(url, {"comments": sample_comments(10)})  # échauffement
    result["latency"] = []
    print(f"  {'batch':>6} | {'total moy. (ms)':>15} | {'total max':>9} | {'modèle (ms)':>11}")
    for size in BATCH_SIZES:
        comments = sample_comments(size)
        totals, server = [], []
        for _ in range(RUNS):
            r, ms = post(url, {"comments": comments})
            r.raise_for_status()
            totals.append(ms)
            server.append(r.json()["processing_time_ms"])
        row = {"batch_size": size, "total_mean_ms": round(statistics.mean(totals), 1),
               "total_max_ms": round(max(totals), 1), "model_mean_ms": round(statistics.mean(server), 1)}
        result["latency"].append(row)
        print(f"  {size:>6} | {row['total_mean_ms']:>15} | {row['total_max_ms']:>9} | {row['model_mean_ms']:>11}")
    return result


def compare(local_url: str, remote_url: str, n: int = 300) -> dict:
    comments = sample_comments(n)
    local = post(local_url, {"comments": comments})[0].json()["predictions"]
    remote = post(remote_url, {"comments": comments})[0].json()["predictions"]
    same_label = sum(a["label"] == b["label"] for a, b in zip(local, remote))
    max_diff = max(abs(a["probabilities"][k] - b["probabilities"][k])
                   for a, b in zip(local, remote) for k in a["probabilities"])
    print(f"\n=== Cohérence local vs déployé ({n} commentaires) ===")
    print(f"  labels identiques : {same_label}/{n}   écart max de probabilité : {max_diff:.6f}")
    return {"n": n, "same_label": same_label, "max_probability_diff": max_diff}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", help="URL de l'API locale (ex. http://localhost:8000)")
    parser.add_argument("--remote", help="URL de l'API déployée")
    args = parser.parse_args()
    if not (args.local or args.remote):
        parser.error("fournir --local et/ou --remote")

    report = {}
    for name, url in (("local", args.local), ("remote", args.remote)):
        if url:
            report[name] = check_api(name, url.rstrip("/"))
    if args.local and args.remote:
        report["consistency"] = compare(args.local.rstrip("/"), args.remote.rstrip("/"))

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {REPORT_PATH}")

    ok = all(all(c == 422 for c in r["errors"].values()) for k, r in report.items() if k != "consistency")
    if "consistency" in report:
        ok &= report["consistency"]["same_label"] == report["consistency"]["n"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
