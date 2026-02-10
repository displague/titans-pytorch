import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description = "Check benchmark regression against a baseline JSON file.")
    parser.add_argument("--baseline", type = str, required = True, help = "Path to baseline benchmark JSON.")
    parser.add_argument("--latest", type = str, required = True, help = "Path to latest benchmark JSON.")
    parser.add_argument(
        "--codebook-baseline",
        type = str,
        default = None,
        help = "Optional baseline JSON from benchmark_codebook_transfer.py."
    )
    parser.add_argument(
        "--codebook-latest",
        type = str,
        default = None,
        help = "Optional latest JSON from benchmark_codebook_transfer.py."
    )
    parser.add_argument(
        "--max-loss-regression-pct",
        type = float,
        default = 5.0,
        help = "Maximum allowed percentage increase for lower-is-better loss metrics."
    )
    parser.add_argument(
        "--max-time-regression-pct",
        type = float,
        default = 10.0,
        help = "Maximum allowed percentage increase for timing metrics."
    )
    parser.add_argument(
        "--min-gain-regression-pct",
        type = float,
        default = 5.0,
        help = "Maximum allowed percentage decrease for higher-is-better gain metrics."
    )
    return parser.parse_args()


def get_path(data, dotted):
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing metric path: {dotted}")
        node = node[key]
    if not isinstance(node, (int, float)):
        raise TypeError(f"Metric path is not numeric: {dotted}")
    return float(node)


def pct_delta(latest, baseline):
    denom = abs(baseline) if abs(baseline) > 1e-12 else 1e-12
    return ((latest - baseline) / denom) * 100.0


def evaluate_metric(name, baseline, latest, direction, threshold_pct):
    delta = pct_delta(latest, baseline)
    passed = True

    if direction == "lower_better":
        passed = delta <= threshold_pct
    elif direction == "higher_better":
        passed = delta >= -threshold_pct
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return dict(
        name = name,
        baseline = baseline,
        latest = latest,
        delta_pct = delta,
        direction = direction,
        threshold_pct = threshold_pct,
        passed = passed
    )


def main():
    args = parse_args()
    baseline_path = Path(args.baseline)
    latest_path = Path(args.latest)

    baseline = json.loads(baseline_path.read_text(encoding = "utf-8"))
    latest = json.loads(latest_path.read_text(encoding = "utf-8"))

    metric_specs = [
        (
            "timing.fwd_ms.symplectic_phase_paging",
            "timing.fwd_ms.symplectic_phase_paging",
            "lower_better",
            args.max_time_regression_pct
        ),
        (
            "timing.train_ms.symplectic_phase_paging",
            "timing.train_ms.symplectic_phase_paging",
            "lower_better",
            args.max_time_regression_pct
        ),
        (
            "spiral.symplectic_phase_paging",
            "spiral.symplectic_phase_paging",
            "lower_better",
            args.max_loss_regression_pct
        ),
        (
            "helix_drift.symplectic_phase_paging",
            "helix_drift.symplectic_phase_paging",
            "lower_better",
            args.max_loss_regression_pct
        ),
        (
            "long_horizon_recovery.symplectic_phase_paging.clean_mse_post",
            "long_horizon_recovery.symplectic_phase_paging.clean_mse_post",
            "lower_better",
            args.max_loss_regression_pct
        ),
        (
            "long_horizon_recovery.symplectic_phase_paging.phase_recovery_gain",
            "long_horizon_recovery.symplectic_phase_paging.phase_recovery_gain",
            "higher_better",
            args.min_gain_regression_pct
        )
    ]

    results = []
    for name, path, direction, threshold in metric_specs:
        baseline_value = get_path(baseline, path)
        latest_value = get_path(latest, path)
        results.append(
            evaluate_metric(
                name = name,
                baseline = baseline_value,
                latest = latest_value,
                direction = direction,
                threshold_pct = threshold
            )
        )

    if (args.codebook_baseline is None) ^ (args.codebook_latest is None):
        raise ValueError("Both --codebook-baseline and --codebook-latest must be provided together.")

    if args.codebook_baseline and args.codebook_latest:
        codebook_baseline = json.loads(Path(args.codebook_baseline).read_text(encoding = "utf-8"))
        codebook_latest = json.loads(Path(args.codebook_latest).read_text(encoding = "utf-8"))
        codebook_metrics = [
            (
                "codebook_transfer.clean_mse_post",
                "variants.codebook_champion_paging.long_horizon.clean_mse_post",
                "lower_better",
                args.max_loss_regression_pct
            ),
            (
                "codebook_transfer.phase_err_post",
                "variants.codebook_champion_paging.long_horizon.phase_err_post",
                "lower_better",
                args.max_loss_regression_pct
            ),
            (
                "codebook_transfer.interference_post_a",
                "variants.codebook_champion_paging.interference.post_a_loss",
                "lower_better",
                args.max_loss_regression_pct
            ),
            (
                "codebook_transfer.transfer_score",
                "variants.codebook_champion_paging.transfer_score",
                "lower_better",
                args.max_loss_regression_pct
            )
        ]
        for name, path, direction, threshold in codebook_metrics:
            baseline_value = get_path(codebook_baseline, path)
            latest_value = get_path(codebook_latest, path)
            results.append(
                evaluate_metric(
                    name = name,
                    baseline = baseline_value,
                    latest = latest_value,
                    direction = direction,
                    threshold_pct = threshold
                )
            )

    print("Regression Check")
    print("-" * 88)
    print(f"{'Metric':52s} | {'Baseline':>10s} | {'Latest':>10s} | {'Delta%':>8s} | Status")
    print("-" * 88)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['name']:52s} | "
            f"{r['baseline']:10.6f} | "
            f"{r['latest']:10.6f} | "
            f"{r['delta_pct']:8.2f} | "
            f"{status}"
        )

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("-" * 88)
        print("Regression guard failed.")
        raise SystemExit(1)

    print("-" * 88)
    print("Regression guard passed.")


if __name__ == "__main__":
    main()
