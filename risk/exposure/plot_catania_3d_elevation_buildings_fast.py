#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

ENV_DIR = Path(sys.executable).resolve().parent.parent
ENV_BIN = ENV_DIR / "bin"
ENV_PROJ = ENV_DIR / "share" / "proj"
os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"
if ENV_PROJ.exists():
    os.environ.setdefault("PROJ_LIB", str(ENV_PROJ))
    os.environ.setdefault("PROJ_DATA", str(ENV_PROJ))
os.environ.setdefault("GMT_USERDIR", "/tmp/gmt_pygmt_catania_fast_3d")
Path(os.environ["GMT_USERDIR"]).mkdir(parents=True, exist_ok=True)

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from shapely.geometry import box


MLDIR = Path(os.environ.get("MLDir", "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami"))
GRID = MLDIR / "data/processed/CT_defbathy.nc"
GPKG = MLDIR / "risk/exposure/CT_Building_Footprint.gpkg"
TERRAIN_CPT = MLDIR / "scripts/PaperIIPlots/PaperI/r2/bathy_error.cpt"
WORK = Path("./catania_fast_3d_blocks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast PyGMT 3D Catania elevation surface with simple building blocks."
    )
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--gpkg", type=Path, default=GPKG)
    parser.add_argument("--out", type=Path, default=Path("/tmp/catania_3d_elevation_building_blocks.png"))
    parser.add_argument("--max-buildings", type=int, default=int(os.environ.get("MAX_BUILDINGS", "2500")))
    parser.add_argument("--dem-stride", type=int, default=int(os.environ.get("DEM_STRIDE", "3")))
    parser.add_argument("--height-scale", type=float, default=float(os.environ.get("HEIGHT_SCALE", "3.0")))
    parser.add_argument("--azimuth", type=float, default=float(os.environ.get("VIEW_AZIMUTH", "140")))
    parser.add_argument("--elevation", type=float, default=float(os.environ.get("VIEW_ELEVATION", "23")))
    parser.add_argument("--projection", default=os.environ.get("PROJECTION", "M18c"))
    parser.add_argument("--zsize", default=os.environ.get("ZSIZE", "5c"))
    parser.add_argument("--terrain-cpt", type=Path, default=TERRAIN_CPT)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*WAL-enabled database.*", category=RuntimeWarning)
        return gpd.read_file(path, layer=layer)


def finite_bounds(bounds: np.ndarray) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = [float(value) for value in bounds]
    if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
        raise ValueError("Invalid bounds")
    return xmin, xmax, ymin, ymax


def intersect_regions(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    xmin = max(a[0], b[0])
    xmax = min(a[1], b[1])
    ymin = max(a[2], b[2])
    ymax = min(a[3], b[3])
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("DEM and building footprints do not overlap")
    return xmin, xmax, ymin, ymax


def pad_region(region: tuple[float, float, float, float], fraction: float = 0.035) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = region
    dx = (xmax - xmin) * fraction
    dy = (ymax - ymin) * fraction
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def crop_dem(grid_path: Path, region: tuple[float, float, float, float], stride: int) -> xr.DataArray:
    ds = xr.open_dataset(grid_path)
    z = ds["z"]
    xmin, xmax, ymin, ymax = region
    dem = z.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
    if stride > 1:
        dem = dem.isel(x=slice(None, None, stride), y=slice(None, None, stride))
    return dem.load()


def sample_ground(full_dem: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    sampled = full_dem.interp(
        x=xr.DataArray(lon, dims="points"),
        y=xr.DataArray(lat, dims="points"),
        kwargs={"fill_value": np.nan},
    ).to_numpy()
    return sampled.astype(float)


def load_buildings(gpkg: Path, dem: xr.DataArray, region: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    footprints = read_layer(gpkg, "Building_Footprint")
    attrs = read_layer(gpkg, "Building_ExposureInfo")
    attrs = attrs[["IDAG", "Elevation", "GHSL_NH", "nFloors"]].copy()

    footprints["IDAG"] = footprints["IDAG"].astype(str)
    attrs["IDAG"] = attrs["IDAG"].astype(str)
    buildings = footprints.merge(attrs, on="IDAG", how="left")

    xmin, xmax, ymin, ymax = region
    region_box = box(xmin, ymin, xmax, ymax)
    buildings = buildings[buildings.intersects(region_box)].copy()

    centroids = buildings.geometry.representative_point()
    ground = sample_ground(dem, centroids.x.to_numpy(), centroids.y.to_numpy())
    exposure_ground = pd.to_numeric(buildings["Elevation"], errors="coerce").to_numpy(dtype=float)
    ground = np.where(np.isfinite(ground), ground, exposure_ground)
    buildings["ground_m"] = np.where(np.isfinite(ground), ground, 0.0)

    ghsl_height = pd.to_numeric(buildings["GHSL_NH"], errors="coerce").to_numpy(dtype=float)
    floor_height = pd.to_numeric(buildings["nFloors"], errors="coerce").fillna(1).to_numpy(dtype=float) * 3.0
    height = np.where(np.isfinite(ghsl_height) & (ghsl_height > 0), ghsl_height, floor_height)
    buildings["height_m"] = np.where(np.isfinite(height) & (height > 0), height, 3.0)
    buildings["rank"] = buildings["PlinthArea"].fillna(1).astype(float) * np.clip(buildings["height_m"], 1, 50)
    return buildings


def select_buildings(gdf: gpd.GeoDataFrame, max_count: int) -> gpd.GeoDataFrame:
    if max_count <= 0 or len(gdf) <= max_count:
        return gdf.copy()

    selected = gdf.copy()
    points = selected.geometry.representative_point()
    xmin, xmax, ymin, ymax = finite_bounds(selected.total_bounds)
    nx, ny = 36, 22
    selected["cell_x"] = np.clip(((points.x - xmin) / (xmax - xmin) * nx).astype(int), 0, nx - 1)
    selected["cell_y"] = np.clip(((points.y - ymin) / (ymax - ymin) * ny).astype(int), 0, ny - 1)

    ranked = selected.sort_values("rank", ascending=False)
    occupied = max(1, ranked.groupby(["cell_x", "cell_y"]).ngroups)
    per_cell = max(1, int(np.ceil(max_count / occupied)))
    balanced = ranked.groupby(["cell_x", "cell_y"], group_keys=False).head(per_cell)
    if len(balanced) > max_count:
        balanced = balanced.nlargest(max_count, "rank")
    elif len(balanced) < max_count:
        rest = ranked.loc[~ranked.index.isin(balanced.index)].head(max_count - len(balanced))
        balanced = pd.concat([balanced, rest], axis=0)
    return gpd.GeoDataFrame(balanced, geometry="geometry", crs=gdf.crs).copy()


def truncate_up(value: float, base: int = 10) -> int:
    return int(math.ceil(float(value) / base) * base)


def write_block_segments(gdf: gpd.GeoDataFrame, out_dir: Path, height_scale: float) -> tuple[Path, Path, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    roof_path = out_dir / "catania_building_block_roofs.gmt"
    side_path = out_dir / "catania_building_block_sides.gmt"
    zmax = 0.0

    with roof_path.open("w", encoding="utf-8") as roofs, side_path.open("w", encoding="utf-8") as sides:
        for row in gdf.itertuples():
            geom = row.geometry.envelope
            coords = list(geom.exterior.coords)
            if len(coords) < 4:
                continue
            base = float(row.ground_m)
            top = base + float(row.height_m) * height_scale
            zmax = max(zmax, top)

            roofs.write(">\n")
            for x, y in coords:
                roofs.write(f"{x:.8f} {y:.8f} {top:.3f}\n")

            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                sides.write(">\n")
                sides.write(f"{x1:.8f} {y1:.8f} {base:.3f}\n")
                sides.write(f"{x2:.8f} {y2:.8f} {base:.3f}\n")
                sides.write(f"{x2:.8f} {y2:.8f} {top:.3f}\n")
                sides.write(f"{x1:.8f} {y1:.8f} {top:.3f}\n")
                sides.write(f"{x1:.8f} {y1:.8f} {base:.3f}\n")

    return roof_path, side_path, zmax


def main() -> None:
    args = parse_args()

    full_dem = xr.open_dataset(args.grid)["z"].load()
    grid_region = (
        float(full_dem.x.min()),
        float(full_dem.x.max()),
        float(full_dem.y.min()),
        float(full_dem.y.max()),
    )

    footprints = read_layer(args.gpkg, "Building_Footprint")
    building_region = finite_bounds(footprints.total_bounds)
    region = pad_region(intersect_regions(grid_region, building_region))
    dem = crop_dem(args.grid, region, max(1, args.dem_stride))

    buildings = load_buildings(args.gpkg, full_dem, region)
    buildings = select_buildings(buildings, args.max_buildings)
    roof_path, side_path, building_zmax = write_block_segments(buildings, WORK, args.height_scale)

    zmin = float(np.nanmin(dem.to_numpy()))
    terrain_zmax = float(np.nanmax(dem.to_numpy()))
    zmax = truncate_up(max(terrain_zmax, building_zmax), base=10)
    region3d = [region[0], region[1], region[2], region[3], math.floor(zmin / 10) * 10, zmax]

    print(f"DEM: {args.grid}")
    print(f"Buildings: {args.gpkg}")
    print(f"Plotting {len(buildings):,} building blocks")
    print(f"Region: {region3d}")
    print(f"Block files: {side_path}, {roof_path}")
    if args.prepare_only:
        return

    perspective = [args.azimuth, args.elevation]
    min_ele = region3d[4]
    max_ele = region3d[5]
    fig = pygmt.Figure()
    with pygmt.config(FORMAT_GEO_MAP="ddd.xxF", MAP_FRAME_TYPE="fancy", FONT="22p"):
        cmap = pygmt.makecpt(cmap=str(args.terrain_cpt), continuous=False, overrule_bg=True)
        fig.grdview(
            grid=dem,
            region=region3d,
            projection=args.projection,
            zsize=args.zsize,
            perspective=perspective,
            surftype="c",
            cmap=True,
            frame=["z100+lmeters", "wSEnZ"],
            plane=f"{min_ele}+ggrey",
            shading="+a270+nt2",
        )
        fig.plot3d(
            data=str(side_path),
            region=region3d,
            projection=args.projection,
            zsize=args.zsize,
            perspective=True,
            fill="219/107/38@25",
            pen="0.03p,120/60/20@35",
            close=True,
        )
        fig.plot3d(
            data=str(roof_path),
            region=region3d,
            projection=args.projection,
            zsize=args.zsize,
            perspective=True,
            fill="255/178/90@8",
            pen="0.04p,120/60/20@25",
            close=True,
        )
        fig.basemap(perspective=True, rose="JCR+w5c+l+o2.5c/0c")

    with pygmt.config(FONT="16p", FONT_ANNOT="18p", FONT_LABEL="20p"):
        fig.colorbar(
            cmap=cmap,
            frame=["a100f20", "x+lElevation (m)"],
            position="JBC+w10c/2c+h+o0.2c/0.2c",
        )

    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
