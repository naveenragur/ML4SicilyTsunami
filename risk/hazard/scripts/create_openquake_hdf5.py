import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openquake.baselib import hdf5
from openquake.hazardlib import site


DEFAULT_NPY_DIR = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/npy_inputs"
)
DEFAULT_OQ_DIR = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/oq_inputs"
)
DEFAULT_SITES_PATH = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/processed/lat_lon_idx_CT_892.txt"
)
VARIANTS = ("mean", "sigmaminus", "sigmaplus")
TRUE_SIZE = "true"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create OpenQuake HDF5 GMF inputs from filtered hazard arrays."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["all"],
        help="Training sizes to process, or 'all' to discover available triplets.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
        help="Prediction statistics to process.",
    )
    parser.add_argument(
        "--npy-dir",
        "--hazard-dir",
        dest="npy_dir",
        type=Path,
        default=DEFAULT_NPY_DIR,
        help="Directory containing filtered .npy triplets.",
    )
    parser.add_argument(
        "--oq-dir",
        "--loss-dir",
        dest="oq_dir",
        type=Path,
        default=DEFAULT_OQ_DIR,
        help="Directory where OpenQuake HDF5 inputs are written.",
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    parser.add_argument("--num-events", type=int, default=53550)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_sizes(hazard_dir, variants):
    sizes = set()
    for path in hazard_dir.glob("pred_d_dask_*_mean.npy"):
        parts = path.stem.split("_")
        if len(parts) >= 4 and parts[3].isdigit():
            sizes.add(parts[3])

    complete_sizes = []
    for size in sizes:
        if all(triplet_exists(hazard_dir, size, variant) for variant in variants):
            complete_sizes.append(size)
    return sorted(complete_sizes, key=int)


def sort_sizes(sizes):
    return sorted(
        sizes,
        key=lambda size: (
            not str(size).isdigit(),
            int(size) if str(size).isdigit() else str(size),
        ),
    )


def is_true_size(size):
    return str(size).lower() == TRUE_SIZE


def triplet_paths(hazard_dir, size, variant):
    if is_true_size(size):
        return {
            "depth": hazard_dir / "true_d_dask.npy",
            "event": hazard_dir / "event_id_true.npy",
            "site": hazard_dir / "site_id_true.npy",
        }

    suffix = f"{size}_{variant}"
    return {
        "depth": hazard_dir / f"pred_d_dask_{suffix}.npy",
        "event": hazard_dir / f"event_id_pred_{suffix}.npy",
        "site": hazard_dir / f"site_id_pred_{suffix}.npy",
    }


def triplet_exists(hazard_dir, size, variant):
    return all(path.exists() for path in triplet_paths(hazard_dir, size, variant).values())


def variants_for_size(size, variants):
    return (None,) if is_true_size(size) else variants


def output_name(size, variant):
    if is_true_size(size):
        return "tsunami_prob_true.hdf5"
    return f"tsunami_prob_{size}_{variant}.hdf5"


def size_label(size, variant):
    if is_true_size(size):
        return TRUE_SIZE
    return f"{size} {variant}"


def build_site_collection(sites_path):
    sites = pd.read_csv(sites_path, sep=",", header=None)
    sites.columns = ["m", "n", "lat", "lon"]
    sites = sites[["lat", "lon"]]
    return site.SiteCollection.from_points(sites["lon"], sites["lat"], req_site_params={})


def write_hdf5(output_path, paths, haz_sitecol, num_events):
    pred_d = np.load(paths["depth"], mmap_mode="r")
    pred_event_id = np.load(paths["event"], mmap_mode="r")
    pred_site_id = np.load(paths["site"], mmap_mode="r")

    if not (len(pred_d) == len(pred_event_id) == len(pred_site_id)):
        raise ValueError(
            f"Mismatched array lengths for {output_path.name}: "
            f"depth={len(pred_d)}, event={len(pred_event_id)}, site={len(pred_site_id)}"
        )

    with hdf5.File(output_path, "w") as hf:
        hf.create_group("gmf_data")
        hf.create_dataset("gmf_data/sid", data=pred_site_id, dtype="uint32")
        hf.create_dataset("gmf_data/eid", data=pred_event_id, dtype="uint32")
        hf.create_dataset("gmf_data/FLOWDEPTH", data=pred_d, dtype="float32")
        hf["gmf_data"].attrs["__pdcolumns__"] = "sid eid FLOWDEPTH"
        hf["gmf_data"].attrs["imts"] = "FLOWDEPTH"
        hf["gmf_data"].attrs["num_events"] = num_events
        hf["gmf_data"].attrs["investigation_time"] = 1
        hf["gmf_data"].attrs["effective_time"] = 1
        hf.create_group("sitecol")
        hf["sitecol"] = haz_sitecol


def main():
    args = parse_args()
    args.oq_dir.mkdir(parents=True, exist_ok=True)
    haz_sitecol = build_site_collection(args.sites_path)

    if args.sizes == ["all"]:
        sizes = discover_sizes(args.npy_dir, args.variants)
    else:
        sizes = sort_sizes(args.sizes)

    if not sizes:
        raise SystemExit("No complete filtered triplets found for the requested variants.")

    for size in sizes:
        for variant in variants_for_size(size, args.variants):
            paths = triplet_paths(args.npy_dir, size, variant)
            missing = [str(path) for path in paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing filtered arrays for {size_label(size, variant)}: {missing}"
                )

            output_path = args.oq_dir / output_name(size, variant)
            if output_path.exists() and not args.overwrite:
                print(f"Skipping {output_path.name}: already exists")
                continue

            print(f"Writing {output_path.name}")
            write_hdf5(output_path, paths, haz_sitecol, args.num_events)
            print(f"HDF5 file created successfully for {size_label(size, variant)}")


if __name__ == "__main__":
    main()
