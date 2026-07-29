'''
Author: Midrar Adham
Created: Wed Jul 29 2026

Converts the remaining 160 static (formerly ieeezipload-driven) triplex_load
objects on feeder S1 to OCHRE-driven HELICS loads, following the exact same
pattern wire_9500_helics.py already applied to the 108 clustered points -
except these are already single-house points (one existing transformer each),
so no transformer relabeling or clustering is needed, just a direct 1:1
building_id -> load conversion.

Run this after wire_9500_helics.py has already converted the 761 clustered
houses. Extends (does not replace) config/9500/load_paths.json; re-run
generate_9500_cosim_config.py afterward to regenerate the HELICS config JSON
with the grown building list.
'''
import json
import os
import re
from pathlib import Path

NON_REAL_TIME_DIR = Path(__file__).parent.parent
MODELS_DIR = NON_REAL_TIME_DIR / "models" / "9500"
CONFIG_DIR = NON_REAL_TIME_DIR / "config" / "9500"
IEEE9500_DOCS_DIR = NON_REAL_TIME_DIR.parent.parent / "ieee9500_docs"

BASE_GLM = MODELS_DIR / "model_base_9500.glm"
HTML_FILE = IEEE9500_DOCS_DIR / "ieee9500_csip.html"
LOAD_PATHS_FILE = CONFIG_DIR / "load_paths.json"
RESSTOCK_DATA_DIR = "/mnt/datasets/resstock_2024/cosimulation"
RESSTOCK_SUBPATH = "up02/simulation_results_august/ochre.parquet"

TARGET_FEEDER = "S1"
NUM_NEEDED = 160


def load_feeder_map(html_file: Path) -> dict:
    html = html_file.read_text()
    start = html.index("const RAW_NODES")
    arr_start = html.index("[", start)
    end_marker = html.index("const RAW_EDGES")
    snippet = html[arr_start:end_marker].rstrip().rstrip(";").rstrip()
    nodes = json.loads(snippet)
    return {n["data"]["id"]: n["data"].get("feeder") for n in nodes}


def find_static_load_candidates(base_text: str, feeder_of: dict) -> list:
    """Remaining (non ochre_house_load_*) triplex_load objects on the target
    feeder, in file order."""
    candidates = []
    for m in re.finditer(r"object triplex_load \{.*?\n\}", base_text, re.S):
        block = m.group(0)
        name_m = re.search(r'name\s+"?([\w.-]+)"?\s*;', block)
        parent_m = re.search(r'parent\s+"?([\w.-]+)"?\s*;', block)
        phases_m = re.search(r"\bphases\s+([\w|]+)\s*;", block)
        volt_m = re.search(r"nominal_voltage\s+([\d.]+)\s*;", block)
        if not (name_m and parent_m and phases_m and volt_m):
            continue
        name = name_m.group(1)
        if name.startswith("ochre_house_load_"):
            continue
        if feeder_of.get(parent_m.group(1)) != TARGET_FEEDER:
            continue
        candidates.append({
            "span": m.span(),
            "name": name,
            "parent": parent_m.group(1),
            "phases": phases_m.group(1),
            "nominal_voltage": volt_m.group(1),
        })
    return candidates


def pick_unused_building_ids(already_used: set, n: int) -> list:
    picked = []
    for bid in sorted(os.listdir(RESSTOCK_DATA_DIR), key=lambda x: int(x)):
        if bid in already_used:
            continue
        parquet_path = Path(RESSTOCK_DATA_DIR) / bid / RESSTOCK_SUBPATH
        if parquet_path.exists():
            picked.append(bid)
        if len(picked) == n:
            break
    return picked


def build_house_load_block(building_id: str, parent_node: str, phases: str, nominal_voltage: str) -> str:
    return (
        "object triplex_load {\n"
        f'  name "ochre_house_load_{building_id}";\n'
        f'  parent "{parent_node}";\n'
        f"  phases {phases};\n"
        f"  nominal_voltage {nominal_voltage};\n"
        "  constant_power_12 0+0j;\n"
        "}"
    )


def main():
    feeder_of = load_feeder_map(HTML_FILE)
    base_text = BASE_GLM.read_text()
    candidates = find_static_load_candidates(base_text, feeder_of)
    if len(candidates) != NUM_NEEDED:
        raise SystemExit(f"Expected {NUM_NEEDED} remaining S1 static loads, found {len(candidates)}")

    with open(LOAD_PATHS_FILE, "r") as f:
        existing_load_paths = json.load(f)
    already_used = {key[len("load_"):] for key in existing_load_paths.keys()}

    building_ids = pick_unused_building_ids(already_used, NUM_NEEDED)
    if len(building_ids) != NUM_NEEDED:
        raise SystemExit(f"Only found {len(building_ids)} unused buildings with valid up02 data, need {NUM_NEEDED}")

    edits = []
    manifest = []
    for candidate, bid in zip(candidates, building_ids):
        new_block = build_house_load_block(
            bid, candidate["parent"], candidate["phases"], candidate["nominal_voltage"]
        )
        edits.append((candidate["span"][0], candidate["span"][1], new_block))
        manifest.append({
            "original_load_name": candidate["name"],
            "parent_node": candidate["parent"],
            "building_id": bid,
            "load_name": f"ochre_house_load_{bid}",
        })
        existing_load_paths[f"load_{bid}"] = f"{RESSTOCK_DATA_DIR}/{bid}/{RESSTOCK_SUBPATH}"

    edits.sort(key=lambda e: e[0])
    pieces = []
    cursor = 0
    for start, end, replacement in edits:
        pieces.append(base_text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(base_text[cursor:])
    new_base_text = "".join(pieces)

    BASE_GLM.write_text(new_base_text)

    with open(LOAD_PATHS_FILE, "w") as f:
        json.dump(existing_load_paths, f, indent=4)

    manifest_file = CONFIG_DIR / "remaining_s1_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Converted {len(candidates)} remaining S1 static loads to OCHRE-driven.")
    print(f"Wrote {BASE_GLM}")
    print(f"Updated {LOAD_PATHS_FILE} (now {len(existing_load_paths)} buildings total)")
    print(f"Wrote {manifest_file}")


if __name__ == "__main__":
    main()
