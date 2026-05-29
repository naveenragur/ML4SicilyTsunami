import argparse
from pathlib import Path

import dask.array as da
import numpy as np


DEFAULT_INPUT_DIR = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/CT/multifoldMC/PTHA"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/npy_inputs"
)

VARIANT_INPUTS = {
    "mean": "pred_d_{size}_direct.npy",
    "sigmaminus": "sigma_minus_{size}_direct.npy",
    "sigmaplus": "sigma_plus_{size}_direct.npy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare filtered OpenQuake hazard arrays for prediction statistics."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["all"],
        help="Training sizes to process, or 'all' to discover complete sizes.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_INPUTS),
        choices=list(VARIANT_INPUTS),
        help="Prediction statistics to process.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output triplets.",
    )
    return parser.parse_args()


def discover_sizes(input_dir, variants):
    sizes = set()
    for path in input_dir.glob("pred_d_*_direct.npy"):
        parts = path.stem.split("_")
        if len(parts) >= 4 and parts[2].isdigit():
            sizes.add(parts[2])

    complete_sizes = []
    for size in sizes:
        if all((input_dir / VARIANT_INPUTS[v].format(size=size)).exists() for v in variants):
            complete_sizes.append(size)
    return sorted(complete_sizes, key=int)


def output_paths(output_dir, size, variant):
    suffix = f"{size}_{variant}"
    return {
        "depth": output_dir / f"pred_d_dask_{suffix}.npy",
        "event": output_dir / f"event_id_pred_{suffix}.npy",
        "site": output_dir / f"site_id_pred_{suffix}.npy",
    }


def outputs_exist(paths):
    return all(path.exists() for path in paths.values())


def save_dask_array(path, array):
    np.save(path, array.compute())


def save_filtered_triplet(input_path, paths, threshold):
    pred_d = np.load(input_path, mmap_mode="r")
    num_events, num_sites = pred_d.shape

    pred_d_dask = da.from_array(pred_d, chunks="auto").astype(np.float16).flatten()
    non_zero_idx = pred_d_dask > threshold

    pred_d_filtered = pred_d_dask[non_zero_idx]
    print(f"{input_path.name}: depths {pred_d_dask.shape} -> {pred_d_filtered.shape}")
    save_dask_array(paths["depth"], pred_d_filtered)
    del pred_d_filtered

    event_id_dask = da.arange(0, num_events, chunks="auto").astype(np.uint16)
    event_id_dask = da.repeat(event_id_dask, num_sites).flatten()
    event_id_filtered = event_id_dask[non_zero_idx]
    print(f"{input_path.name}: event ids -> {event_id_filtered.shape}")
    save_dask_array(paths["event"], event_id_filtered)
    del event_id_dask, event_id_filtered

    site_id_dask = da.arange(0, num_sites, chunks="auto").astype(np.uint32)
    site_id_dask = da.tile(site_id_dask, num_events).flatten()
    site_id_filtered = site_id_dask[non_zero_idx]
    print(f"{input_path.name}: site ids -> {site_id_filtered.shape}")
    save_dask_array(paths["site"], site_id_filtered)
    del site_id_dask, site_id_filtered, pred_d_dask, pred_d


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sizes == ["all"]:
        sizes = discover_sizes(args.input_dir, args.variants)
    else:
        sizes = sorted(args.sizes, key=int)

    if not sizes:
        raise SystemExit("No complete sizes found for the requested variants.")

    for size in sizes:
        for variant in args.variants:
            input_path = args.input_dir / VARIANT_INPUTS[variant].format(size=size)
            if not input_path.exists():
                raise FileNotFoundError(input_path)

            paths = output_paths(args.output_dir, size, variant)
            if outputs_exist(paths) and not args.overwrite:
                print(f"Skipping {size} {variant}: outputs already exist")
                continue

            print(f"Preparing {size} {variant}")
            save_filtered_triplet(input_path, paths, args.threshold)


if __name__ == "__main__":
    main()
