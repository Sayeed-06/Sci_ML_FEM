#!/usr/bin/env python3
"""Run variance-based Sobol analysis for the trained similarity DeepONet.

Six independent physical inputs are sampled over the same observed ranges used
by the project-model analysis.  Each physical sample is transformed to the
as-trained similarity representation [l, phi, P/Q] while plate geometry and the
load footprint are held fixed at a representative dataset case.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.sim_dataset import load_norm_stats
from dataset.sim_graph import build_sim_graph
from models.deeponet import build_sim_deeponet
from parse_in2 import parse_in2
from parse_ou2 import parse_ou2
from similarity_transform import compute_similarity_params


PROBLEM = {
    "num_vars": 6,
    "names": ["E", "thickness", "k_winkler", "delta_T", "alpha_T", "pressure_q"],
    "bounds": [
        [27603.03, 41399.87],
        [150.02, 380.00],
        [0.02708, 0.29999],
        [0.00, 49.99],
        [6.90024626e-06, 1.19998119e-05],
        [0.70, 1.00],
    ],
}

QOI_NAMES = ("mean_abs_w", "median_abs_w", "p95_abs_w", "max_abs_w")


def representative_case(data_root: Path) -> Path:
    lower = np.asarray(PROBLEM["bounds"], dtype=float)[:, 0]
    upper = np.asarray(PROBLEM["bounds"], dtype=float)[:, 1]
    midpoint = (lower + upper) / 2.0
    scale = upper - lower
    best_path, best_distance = None, float("inf")
    for case in sorted(data_root.glob("case_*")):
        input_path = case / "input.in2"
        output_path = case / "input.ou2"
        if not input_path.exists() or not output_path.exists():
            continue
        values = parse_in2(input_path)
        vector = np.asarray([
            values["E"], values["thickness"], values["k_winkler"],
            values["delta_T"], values["alpha_T"], values["pressure_q"],
        ])
        distance = float(np.sum(((vector - midpoint) / scale) ** 2))
        if distance < best_distance:
            best_path, best_distance = case, distance
    if best_path is None:
        raise RuntimeError(f"No FEM cases found under {data_root}")
    return best_path


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    model = build_sim_deeponet(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, checkpoint


def graph_for_values(inp_base: dict, out_base: dict, values: np.ndarray):
    inp = inp_base.copy()
    for name, value in zip(PROBLEM["names"], values):
        inp[name] = float(value)
    sim = compute_similarity_params(
        E=inp["E"], h=inp["thickness"], k_file=inp["k_winkler"],
        mu=inp["nu"], alpha=inp["alpha_T"], gamma_file=inp["gamma"],
        delta_T=inp["delta_T"], pressure_q=inp["pressure_q"],
    )
    return build_sim_graph(inp, out_base, sim, include_diagonals=True), sim


@torch.inference_mode()
def evaluate_samples(model, inp_base, out_base, samples, norm_stats, device, batch_size):
    outputs = {name: np.empty(len(samples), dtype=np.float64) for name in QOI_NAMES}
    gf_mean = norm_stats["gf_mean"].reshape(1, -1)
    gf_std = norm_stats["gf_std"].reshape(1, -1)
    y_mean = norm_stats["y_mean"].to(device)
    y_std = norm_stats["y_std"].to(device)
    for start in range(0, len(samples), batch_size):
        rows = samples[start : start + batch_size]
        graphs = []
        for values in rows:
            graph, _ = graph_for_values(inp_base, out_base, values)
            graph.global_feat = (graph.global_feat - gf_mean) / gf_std
            graphs.append(graph)
        batch = Batch.from_data_list(graphs).to(device)
        prediction = model(batch) * y_std + y_mean
        for local_index in range(len(rows)):
            left = int(batch.ptr[local_index])
            right = int(batch.ptr[local_index + 1])
            absolute_w = prediction[left:right, 0].abs()
            out_index = start + local_index
            outputs["mean_abs_w"][out_index] = float(absolute_w.mean())
            outputs["median_abs_w"][out_index] = float(torch.quantile(absolute_w, 0.50))
            outputs["p95_abs_w"][out_index] = float(torch.quantile(absolute_w, 0.95))
            outputs["max_abs_w"][out_index] = float(absolute_w.max())
        if start == 0 or (start // batch_size + 1) % 100 == 0:
            print(f"  evaluated {min(start + len(rows), len(samples))}/{len(samples)}", flush=True)
    return outputs


def analyze_outputs(outputs, sample_size, num_resamples, seed):
    results, summaries, resamples = {}, [], []
    for qoi, values in outputs.items():
        analysis = sobol_analyze.analyze(
            PROBLEM, values, calc_second_order=False, num_resamples=num_resamples,
            conf_level=0.95, keep_resamples=True, seed=seed, print_to_console=False,
        )
        results[qoi] = analysis
        for index_name in ("S1", "ST"):
            replicate_values = np.asarray(analysis[f"{index_name}_conf_all"])
            for parameter_index, parameter in enumerate(PROBLEM["names"]):
                distribution = replicate_values[:, parameter_index]
                finite = distribution[np.isfinite(distribution)]
                summaries.append({
                    "sample_size": sample_size, "qoi": qoi,
                    "parameter": parameter, "index": index_name,
                    "estimate": float(analysis[index_name][parameter_index]),
                    "bootstrap_mean": float(np.mean(finite)),
                    "bootstrap_median": float(np.median(finite)),
                    "ci_low": float(np.quantile(finite, 0.025)),
                    "ci_high": float(np.quantile(finite, 0.975)),
                    "confidence_half_width": float(analysis[f"{index_name}_conf"][parameter_index]),
                })
                for replicate, value in enumerate(distribution):
                    resamples.append({
                        "sample_size": sample_size, "qoi": qoi,
                        "parameter": parameter, "index": index_name,
                        "replicate": replicate, "value": float(value),
                    })
    return results, summaries, resamples


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "runs/sim_deeponet_v1/config_resolved.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/sim_deeponet_v1/best.pt")
    parser.add_argument("--norm-stats", type=Path, default=ROOT / "runs/sim_deeponet_v1/norm_stats.pt")
    parser.add_argument("--data-root", type=Path, default=ROOT / "generated_cases")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/sobol_similarity")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--num-resamples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, _, checkpoint = load_model(args.config, args.checkpoint, device)
    norm_stats = load_norm_stats(args.norm_stats)
    case_path = representative_case(args.data_root)
    print(f"Representative fixed geometry: {case_path.name}")
    inp_base = parse_in2(case_path / "input.in2")
    out_base = parse_ou2(case_path / "input.ou2")

    all_summaries, final_resamples, final_results = [], [], None
    for sample_size in args.sample_sizes:
        print(f"\nSobol base sample N={sample_size}")
        samples = sobol_sample.sample(
            PROBLEM, sample_size, calc_second_order=False, scramble=True, seed=args.seed,
        )
        outputs = evaluate_samples(
            model, inp_base, out_base, samples, norm_stats, device, args.batch_size,
        )
        results, summaries, resamples = analyze_outputs(
            outputs, sample_size, args.num_resamples, args.seed,
        )
        all_summaries.extend(summaries)
        if sample_size == args.sample_sizes[-1]:
            final_results, final_resamples = results, resamples

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sobol_convergence_summary.csv", all_summaries)
    write_csv(args.output_dir / "sobol_bootstrap_resamples.csv", final_resamples)
    metadata = {
        "model": "sim_deeponet_v1", "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "fixed_geometry_case": case_path.name,
        "input_distributions": "independent uniform over observed training ranges",
        "physical_to_similarity_mapping": "as-trained compute_similarity_params",
        "unit_conversion_caveat": (
            "The existing similarity transform applies conversions that appear inconsistent "
            "with the generated .in2 magnitudes; results characterize the trained checkpoint."
        ),
        "problem": PROBLEM, "sample_sizes": args.sample_sizes,
        "num_resamples": args.num_resamples, "confidence_level": 0.95,
    }
    with (args.output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    final_n = args.sample_sizes[-1]
    for qoi, analysis in final_results.items():
        payload = {
            "names": PROBLEM["names"],
            "S1": np.asarray(analysis["S1"]).tolist(),
            "S1_conf": np.asarray(analysis["S1_conf"]).tolist(),
            "ST": np.asarray(analysis["ST"]).tolist(),
            "ST_conf": np.asarray(analysis["ST_conf"]).tolist(),
            "metadata": {
                "output": qoi, "confidence_level": 0.95,
                "base_sample_size": final_n,
                "status": "computed from as-trained similarity DeepONet",
                "fixed_geometry_case": case_path.name,
            },
        }
        with (args.output_dir / f"{qoi}_sobol.json").open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
    print(f"\nSaved Sobol results to {args.output_dir}")


if __name__ == "__main__":
    main()
