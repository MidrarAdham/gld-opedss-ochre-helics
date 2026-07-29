'''
Author: Midrar Adham
Created: Tue Jul 28 2026

Wires the IEEE 9500-node feeder into the HELICS/OCHRE cosimulation, following
the same pattern already used by models/powerflow_4node.glm.

Inputs:
  - dataset_map/less_than_five_tons_resstock_2024.csv: building_id -> ResStock parquet path
  - dataset_map/method4_representative_trial.csv: 761 building IDs grouped into 108
    clusters, each sized to a 25/50/75 kVA transformer
  - ieee9500_docs/ieee9500_csip.html: embedded node->feeder (S1/S2/S3) classification,
    used to restrict cluster placement to feeder S1

For each of the first 108 existing single-load S1 residential transformer points
(in file order): relabel its transformer_configuration to the cluster's chosen
size, remove the single existing triplex_load there, and add one triplex_load per
building in the cluster (named ochre_house_load_<id>, constant_power_12 0+0j) so
HELICS subscriptions can drive it directly.
'''
import ast
import csv
import json
import re
import sys
from pathlib import Path

non_real_time_dir = Path(__file__).parent.parent
models_dir = non_real_time_dir / "models"
config_dir = non_real_time_dir / "config"
dataset_map_dir = non_real_time_dir / "dataset_map"
ieee9500_docs_dir = non_real_time_dir.parent.parent / "ieee9500_docs"

base_glm = models_dir / "model_base_9500.glm"
startup_glm = models_dir / "model_startup_9500.glm"
html_file = ieee9500_docs_dir / "ieee9500_csip.html"
resstock_csv = dataset_map_dir / "less_than_five_tons_resstock_2024.csv"
clusters_csv = dataset_map_dir / "method4_representative_trial.csv"

residential_configs = {
"xcon_ct5", "xcon_ct10", "xcon_ct15", "xcon_ct25", "xcon_ct37",
"xcon_ct50", "xcon_ct75", "xcon_ct100", "xcon_ct150", "xcon_ct250",
}
kva_to_config = {25.0: "xcon_ct25", 50.0: "xcon_ct50", 75.0: "xcon_ct75"}
target_feeder = "S1"
num_clusters_needed = 108



def load_feeder_map(html_file: Path) -> dict:
    html = html_file.read_text()
    start = html.index("const RAW_NODES")
    arr_start = html.index("[", start)
    end_marker = html.index("const RAW_EDGES")
    snippet = html[arr_start:end_marker].rstrip().rstrip(";").rstrip()
    nodes = json.loads(snippet)
    return {n["data"]["id"]: n["data"].get("feeder") for n in nodes}


def load_clusters(clusters_csv: Path) -> list:
    clusters = []
    with open(clusters_csv, "r") as f:
        for row in csv.DictReader(f):
            clusters.append({
                "cluster_index": int(row["cluster_index"]),
                "chosen_transformer_kva": float(row["chosen_transformer_kva"]),
                "building_ids": ast.literal_eval(row["building_IDs"]),
            })
    clusters.sort(key=lambda c: c["cluster_index"])
    return clusters


def load_resstock_paths(resstock_csv: Path) -> dict:
    paths = {}
    with open(resstock_csv, "r") as f:
        for row in csv.DictReader(f):
            paths[row["building_id"]] = row["timeseries_file"]
    return paths


def find_transformer_candidates(base_text: str, feeder_of: dict) -> list:
    """Residential-scale transformer objects on the target feeder, in file order."""
    candidates = []
    for m in re.finditer(r"object transformer \{.*?\n\}", base_text, re.S):
        block = m.group(0)
        cfg_m = re.search(r'configuration\s+"?([\w.-]+)"?\s*;', block)
        to_m = re.search(r'\bto\s+"?([\w.-]+)"?\s*;', block)
        if not (cfg_m and to_m):
            continue
        if cfg_m.group(1) not in residential_configs:
            continue
        if feeder_of.get(to_m.group(1)) != target_feeder:
            continue
        candidates.append({
            "span": m.span(),
            "block": block,
            "cfg_span": (m.start() + cfg_m.start(1), m.start() + cfg_m.end(1)),
            "to_node": to_m.group(1),
        })
    return candidates


def find_triplex_line_from_to(base_text: str) -> dict:
    """Each residential transformer's secondary node connects via a triplex_line
    to the actual service node a triplex_load parents to (transformer.to !=
    triplex_load.parent in this feeder export)."""
    from_to = {}
    for m in re.finditer(r"object triplex_line \{.*?\n\}", base_text, re.S):
        block = m.group(0)
        f_m = re.search(r'\bfrom\s+"?([\w.-]+)"?\s*;', block)
        t_m = re.search(r'\bto\s+"?([\w.-]+)"?\s*;', block)
        if f_m and t_m:
            from_to[f_m.group(1)] = t_m.group(1)
    return from_to


def find_triplex_loads_by_parent(base_text: str) -> dict:
    by_parent = {}
    for m in re.finditer(r"object triplex_load \{.*?\n\}", base_text, re.S):
        block = m.group(0)
        parent_m = re.search(r'parent\s+"?([\w.-]+)"?\s*;', block)
        phases_m = re.search(r"\bphases\s+([\w|]+)\s*;", block)
        volt_m = re.search(r"nominal_voltage\s+([\d.]+)\s*;", block)
        if not (parent_m and phases_m and volt_m):
            continue
        by_parent[parent_m.group(1)] = {
            "span": m.span(),
            "phases": phases_m.group(1),
            "nominal_voltage": volt_m.group(1),
        }
    return by_parent


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
    feeder_of = load_feeder_map(html_file)
    clusters = load_clusters(clusters_csv)
    resstock_paths = load_resstock_paths(resstock_csv)

    assert len(clusters) == num_clusters_needed, f"expected {num_clusters_needed} clusters, got {len(clusters)}"
    total_buildings = sum(len(c["building_ids"]) for c in clusters)

    base_text = base_glm.read_text()
    candidates = find_transformer_candidates(base_text, feeder_of)
    if len(candidates) < num_clusters_needed:
        sys.exit(f"Only found {len(candidates)} S1 residential transformer candidates, need {num_clusters_needed}")
    chosen = candidates[:num_clusters_needed]

    loads_by_parent = find_triplex_loads_by_parent(base_text)
    line_from_to = find_triplex_line_from_to(base_text)

    edits = []  # (start, end, replacement_text)
    manifest = []
    all_building_ids = []

    for candidate, cluster in zip(chosen, clusters):
        to_node = candidate["to_node"]
        service_node = line_from_to[to_node]
        new_cfg = kva_to_config[cluster["chosen_transformer_kva"]]

        # 1) relabel the transformer's configuration in place
        cfg_start, cfg_end = candidate["cfg_span"]
        edits.append((cfg_start, cfg_end, new_cfg))

        # 2) replace the single existing triplex_load at the service node
        #    (reached via the transformer's secondary and its triplex_line)
        #    with one load per building in the cluster
        existing_load = loads_by_parent[service_node]
        new_blocks = "\n".join(
            build_house_load_block(bid, service_node, existing_load["phases"], existing_load["nominal_voltage"])
            for bid in cluster["building_ids"]
        )
        edits.append((existing_load["span"][0], existing_load["span"][1], new_blocks))

        manifest.append({
            "cluster_index": cluster["cluster_index"],
            "transformer_to_node": to_node,
            "service_node": service_node,
            "transformer_kva": cluster["chosen_transformer_kva"],
            "transformer_configuration": new_cfg,
            "building_ids": cluster["building_ids"],
            "load_names": [f"ochre_house_load_{bid}" for bid in cluster["building_ids"]],
        })
        all_building_ids.extend(cluster["building_ids"])

    assert len(all_building_ids) == total_buildings == len(set(all_building_ids))

    edits.sort(key=lambda e: e[0])
    for a, b in zip(edits, edits[1:]):
        assert a[1] <= b[0], f"overlapping edits at {a} / {b}"

    pieces = []
    cursor = 0
    for start, end, replacement in edits:
        pieces.append(base_text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(base_text[cursor:])
    new_base_text = "".join(pieces)

    base_glm.write_text(new_base_text)

    manifest_file = config_dir / "model_base_9500_cluster_manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    load_paths_9500 = {f"load_{bid}": resstock_paths[bid] for bid in all_building_ids}
    with open(config_dir / "load_paths_9500.json", "w") as f:
        json.dump(load_paths_9500, f, indent=4)

    print(f"Converted {len(chosen)} S1 transformer points, {len(all_building_ids)} houses total.")
    print(f"Wrote {base_glm}")
    print(f"Wrote {manifest_file}")
    print(f"Wrote {config_dir / 'load_paths_9500.json'}")
    return all_building_ids


if __name__ == "__main__":
    main()
