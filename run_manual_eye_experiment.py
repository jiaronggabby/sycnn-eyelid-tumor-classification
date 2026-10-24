"""New manually cropped SY-CNN experiment; never fall back to old data.

Author: Jia Rong. Dry-run performs data/configuration checks only. Real runs
require CUDA, an explicit GPU, and a separate output directory on the server.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BUNDLE = Path(__file__).resolve().parent


def seed_gpu_plan(seed_values, gpu_devices):
    if not gpu_devices:
        return {}
    return {gpu: [seed for index, seed in enumerate(seed_values) if index % len(gpu_devices) == slot]
            for slot, gpu in enumerate(gpu_devices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", help="Comma-separated physical CUDA devices; one seed runs per listed device")
    parser.add_argument("--stage", choices=["all", "validation", "test", "summary"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gan-batch-size", type=int)
    parser.add_argument("--diffusion-batch-size", type=int)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    config = json.loads((BUNDLE / "manual_eye_training_config.json").read_text(encoding="utf-8"))
    env = dict(config["environment"])
    for argument, key in [(args.gan_batch_size, "GAN_BATCH_SIZE"), (args.diffusion_batch_size, "DIFFUSION_BATCH_SIZE")]:
        if argument is not None:
            if argument < 1:
                raise ValueError("Batch size must be positive")
            env[key] = str(argument)
    env.update(MANUAL_EYE_DATASET_ROOT=str(root), CANONICAL_IMAGE_ROOT=str(root),
               MANIFEST_PATH=str(root / "metadata/dataset_manifest.csv"),
               LOCKED_SPLIT=str(root / "splits/split_definition.json"),
               RETROSPECTIVE_SHARED_PLAN_SEED=str(config["seeds"][0]),
               REQUIRE_CUDA="0" if args.dry_run else "1")
    gpu_devices = []
    if args.gpu is not None:
        gpu_devices = [value.strip() for value in args.gpu.split(",") if value.strip()]
        if not gpu_devices or len(gpu_devices) != len(set(gpu_devices)):
            raise ValueError("--gpu must contain unique comma-separated CUDA device IDs")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    os.environ.update(env)
    if not args.dry_run:
        if args.output_root is None or not gpu_devices:
            raise ValueError("Real execution requires --output-root and --gpu")
        if os.name == "nt":
            raise RuntimeError("Training is server-only; use --dry-run for local preparation")
        output = args.output_root.resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Keep experiment output separate from the dataset tree")
        os.environ["OUTPUT_ROOT"] = str(output)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is disabled")
    from manual_eye_training import load_manual_data
    import reproduce_gan_resnet50_xgb_patient224 as base
    frame, audit, splits = load_manual_data(root, base)
    os.environ["MANUAL_EYE_DATA_VERIFIED"] = "1"
    from full_highres_matrix import build_specs, cohort_defaults
    defaults = cohort_defaults("retrospective_cohort")
    specs = build_specs(defaults["methods"], defaults["backbones"], defaults["heads"],
                        defaults["ratios"], [False], defaults["strategies"], True)
    print(json.dumps({"protocol": config["protocol"], "data": audit,
                      "seeds": config["seeds"], "routes_per_seed": len(specs),
                      "gpu_devices": gpu_devices, "seeds_in_parallel": len(gpu_devices),
                      "seed_gpu_plan": seed_gpu_plan(config["seeds"], gpu_devices),
                      "environment": env, "dry_run": args.dry_run}, indent=2, default=int))
    if args.dry_run:
        return 0
    # Refuse an unrelated/old output directory. Hold one lock across all children.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    source_names = ["full_highres_matrix.py", "reproduce_gan_resnet50_xgb_patient224.py",
                    "generator_quality.py", "manual_eye_training.py", "manual_eye_training_config.json",
                    "run_manual_eye_experiment.py", "run_manual_eye_watchdog.sh",
                    "manual_eye_training_contract.json",
                    "prepare_manual_eye_dataset.py",
                    "requirements.txt",
                    "full_matrix_validate.py", "select_routes.py",
                    "summarize_results.py", "paired_patient_bootstrap.py"]
    record = {"protocol": config["protocol"], "seeds": config["seeds"], "environment": env,
              "sources": {name: base.sha256_file(BUNDLE / name) for name in source_names},
              "split_sha256": base.sha256_file(root / "splits/split_definition.json"),
              "dataset_manifest_sha256": base.sha256_file(root / "metadata/dataset_manifest.csv")}
    fingerprint_record = {**record, "environment": {
        key: value for key, value in env.items() if key != "CUDA_VISIBLE_DEVICES"
    }}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_record, sort_keys=True).encode()).hexdigest()
    marker = output / "manual_eye_experiment.json"
    with (output / "manual_eye_experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous["fingerprint"] != fingerprint:
                raise RuntimeError("Dataset/code/settings differ from this output root; preserve it and use a new root")
        else:
            allowed_root_entries = {"manual_eye_experiment.lock", "manual_eye_watchdog.lock", "logs"}
            unexpected = [path.name for path in output.iterdir() if path.name not in allowed_root_entries]
            logs_path = output / "logs"
            if logs_path.exists():
                if not logs_path.is_dir():
                    unexpected.append("logs (not a directory)")
                else:
                    unexpected.extend(
                        f"logs/{path.name}" for path in logs_path.iterdir()
                        if not path.is_file() or not (
                            path.name == "manual_eye_watchdog_state.log"
                            or path.name.startswith("watchdog_attempt_")
                            and path.suffix in {".log", ".pgid"}
                        )
                    )
            if unexpected:
                raise RuntimeError("Refusing to mix with an existing unrelated experiment output")
            marker.write_text(json.dumps({**record, "fingerprint": fingerprint}, indent=2), encoding="utf-8")
        completion = output / "manual_eye_experiment_complete.json"
        if args.stage == "all" and completion.is_file():
            completed = json.loads(completion.read_text(encoding="utf-8"))
            if completed.get("fingerprint") != fingerprint:
                raise RuntimeError("Completion marker does not match this run configuration")
            print(f"Experiment already complete: {completion}")
            return 0
        logs = output / "logs"
        logs.mkdir(exist_ok=True)

        def run(script, arguments, name, gpu=None):
            with (logs / f"{name}.log").open("a", encoding="utf-8") as stream:
                child_env = dict(os.environ)
                if gpu is not None:
                    child_env["CUDA_VISIBLE_DEVICES"] = gpu
                process = subprocess.Popen([sys.executable, "-B", "-u", str(BUNDLE / script), *map(str, arguments)],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace", env=child_env)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait():
                    raise RuntimeError(f"Stage failed; resume from its existing output: {name}")

        def run_seeds(script, stage, seed_values, extra_arguments=()):
            if not gpu_devices:
                raise RuntimeError("Seed execution requires explicitly assigned GPUs")

            def run_device(gpu, assigned_seeds):
                for seed in assigned_seeds:
                    arguments = ["--seed", seed, "--include-strong", "--stage", stage, *extra_arguments]
                    run(script, arguments, f"seed{seed}_{stage}", gpu=gpu)

            with ThreadPoolExecutor(max_workers=len(gpu_devices)) as executor:
                futures = [executor.submit(run_device, gpu, assigned)
                           for gpu, assigned in seed_gpu_plan(seed_values, gpu_devices).items()]
                for future in as_completed(futures):
                    future.result()

        seeds = ",".join(map(str, config["seeds"]))
        if args.stage in {"all", "validation"}:
            run_seeds("full_highres_matrix.py", "validation", config["seeds"])
        if args.stage in {"all", "test"}:
            run("select_routes.py", [output, "--expected-seeds", seeds], "selection")
            run_seeds("full_highres_matrix.py", "test", config["seeds"],
                      ["--selection-lock", output / "selection_lock.json"])
        if args.stage in {"all", "summary"}:
            run("full_matrix_validate.py", [output, "--expected-seeds", seeds, "--selection-lock",
                                            output / "selection_lock.json", "--bootstrap", "--n-bootstrap", 1000], "statistics")
            for split in ["test1", "source_holdout"]:
                run("summarize_results.py", [output, "--test-split", split], f"summary_{split}")
                run("paired_patient_bootstrap.py", [output, "--test-split", split, "--n-bootstrap", 1000], f"paired_{split}")
        if args.stage == "all":
            temporary = completion.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "status": "complete",
                "fingerprint": fingerprint,
                "completed_seeds": config["seeds"],
                "completed_test_splits": ["test1", "source_holdout"],
            }, indent=2), encoding="utf-8")
            temporary.replace(completion)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
