import argparse
import json
from pathlib import Path

from tools.common import ROOT
from tools.train import decisions, fit_classifier, load_dataset, negative_overlap, scores


def compare(dataset, output):
    labels, splits, _ = load_dataset(dataset)
    report = {}
    for name, columns in [("mfcc_only", slice(0, 30)), ("hybrid", slice(None))]:
        train_x, train_y, _ = splits["train"]
        val_x, val_y, _ = splits["val"]
        test_x, test_y, _ = splits["test"]
        scaler, model, choice, _, _ = fit_classifier(
            train_x[:, columns], train_y, val_x[:, columns], val_y, labels
        )
        threshold, margin = choice["threshold"], choice["margin"]
        test_probs = model.predict_proba(scaler.transform(test_x[:, columns]))
        report[name] = {
            "threshold": threshold,
            "margin": margin,
            "test": scores(
                test_y, decisions(test_x, test_probs, 0, threshold, margin, -3.5), labels, 0
            ),
        }
    report["method"] = (
        "Mesmas consultas, classes, regressão logística e escolha de limiares pela validação. "
    )
    report["method"] += (
        "A diferença é acrescentar duas medidas de alinhamento por título do catálogo às 30 features acústicas."
    )
    report["negative_split_overlap"] = negative_overlap(splits)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara MFCC e fingerprints sem mudar as consultas de teste"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/ablacao.json")
    args = parser.parse_args()
    compare(args.data, args.output)
