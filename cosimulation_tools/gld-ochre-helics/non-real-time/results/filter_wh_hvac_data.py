'''
Author: Midrar Adham
Created: Sat Aug 01 2026
'''
import os
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq



wh_column = "Water Heating Electric Power (kW)"
hvac_column = "HVAC Cooling Electric Power (kW)"


def write_manifest(manifest_filename: str | Path) -> None:
    path_dir = Path("/mnt/datasets/resstock_2024/cosimulation/")
    upgrade = "up02"
    existing_files = []

    for bldg_id in os.listdir(path_dir):
        parquet_file = (path_dir / bldg_id / upgrade / "simulation_results_august" / "ochre.parquet")

        if parquet_file.is_file() and parquet_file.stat().st_size > 0:
            existing_files.append({"path": str(parquet_file)})

    manifest_df = pd.DataFrame(existing_files, columns=["path"])
    manifest_df.to_csv(manifest_filename, index=False)

    print(
        f"Wrote {len(manifest_df)} valid Parquet paths "
        f"to {manifest_filename}"
    )


def has_wh_without_hvac(file_cols: list[str]) -> bool:
    return wh_column in file_cols and hvac_column not in file_cols


def has_hvac(file_cols: list[str]) -> bool:
    return hvac_column in file_cols


def write_wh_hvac_manifests(full_df: pd.DataFrame) -> None:
    wh_manifest_filename = Path("./wh_cosim/wh_manifest.csv")
    hvac_manifest_filename = Path("./hvac_cosim/hvac_manifest.csv")

    wh_manifest_filename.parent.mkdir(parents=True, exist_ok=True)
    hvac_manifest_filename.parent.mkdir(parents=True, exist_ok=True)

    wh_buildings = []
    hvac_buildings = []

    for row in full_df.itertuples(index=False):
        try:
            parquet_file = pq.ParquetFile(row.path)
            file_cols = parquet_file.schema.names

            if has_wh_without_hvac(file_cols):
                wh_buildings.append({"path": row.path})

            if has_hvac(file_cols):
                hvac_buildings.append({"path": row.path})

        except Exception as exc:
            print(f"Could not inspect {row.path}: {exc}")

    wh_df = pd.DataFrame(wh_buildings, columns=["path"])
    hvac_df = pd.DataFrame(hvac_buildings, columns=["path"])

    # wh_df.to_csv(wh_manifest_filename, index=False)
    # hvac_df.to_csv(hvac_manifest_filename, index=False)

    print("There are:")
    print(f"{len(wh_df)} files with WH but no HVAC cooling")
    print(f"{len(hvac_df)} files with HVAC cooling")


def main() -> None:
    manifest_filename = Path("./datasets_manifest.csv")

    if not manifest_filename.is_file():
        write_manifest(manifest_filename)

    full_df = pd.read_csv(manifest_filename)
    write_wh_hvac_manifests(full_df)


if __name__ == "__main__":
    main()