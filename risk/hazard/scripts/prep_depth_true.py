import argparse
from pathlib import Path


DEFAULT_INPUT_PATH = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/CT/multifoldMC/PTHA/true_d_53550.npy"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/npy_inputs"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare filtered OpenQuake hazard arrays for true depths."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing true-depth output triplet.",
    )
    return parser.parse_args()


def output_paths(output_dir):
    return {
        "depth": output_dir / "true_d_dask.npy",
        "event": output_dir / "event_id_true.npy",
        "site": output_dir / "site_id_true.npy",
    }


def outputs_exist(paths):
    return all(path.exists() for path in paths.values())


def save_dask_array(path, array):
    import numpy as np

    np.save(path, array.compute())


def save_filtered_triplet(input_path, paths, threshold):
    import dask.array as da
    import numpy as np

    true_d = np.load(input_path, mmap_mode="r")
    num_events, num_sites = true_d.shape

    true_d_dask = da.from_array(true_d, chunks="auto").astype(np.float16).flatten()
    non_zero_idx = true_d_dask > threshold

    true_d_filtered = true_d_dask[non_zero_idx]
    print(f"{input_path.name}: depths {true_d_dask.shape} -> {true_d_filtered.shape}")
    save_dask_array(paths["depth"], true_d_filtered)
    del true_d_filtered

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
    del site_id_dask, site_id_filtered, true_d_dask, true_d


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_path.exists():
        raise FileNotFoundError(args.input_path)

    paths = output_paths(args.output_dir)
    if outputs_exist(paths) and not args.overwrite:
        print("Skipping true: outputs already exist")
        return

    print("Preparing true")
    save_filtered_triplet(args.input_path, paths, args.threshold)


if __name__ == "__main__":
    main()
