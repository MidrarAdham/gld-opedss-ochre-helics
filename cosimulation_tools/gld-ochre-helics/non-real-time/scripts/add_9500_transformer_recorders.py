'''
Author: Midrar Adham
Created: Wed Jul 29 2026

Adds one recorder per OCHRE-driven S1 transformer (268 total: 108 cluster
transformers from wire_9500_helics.py + 160 individually-converted single-house
transformers from wire_9500_remaining_s1.py), recording power_in/power_out at
60s - mirrors the residential_transformer recorder in powerflow_4node.glm, but
skips per-house recorders since those would just echo what OCHRE already
publishes.

Run once, after both wire_9500_helics.py and wire_9500_remaining_s1.py.
'''
import json
import re
from pathlib import Path

NON_REAL_TIME_DIR = Path(__file__).parent.parent
MODELS_DIR = NON_REAL_TIME_DIR / "models" / "9500"
CONFIG_DIR = NON_REAL_TIME_DIR / "config" / "9500"

BASE_GLM = MODELS_DIR / "model_base_9500.glm"
CLUSTER_MANIFEST = CONFIG_DIR / "cluster_manifest.json"
REMAINING_MANIFEST = CONFIG_DIR / "remaining_s1_manifest.json"
RESULTS_DIR_REL = "../../results/9500_transformers"


def build_transformer_name_map(base_text: str) -> dict:
    """transformer 'to' node -> transformer object name, for all transformers."""
    name_of = {}
    for m in re.finditer(r"object transformer \{.*?\n\}", base_text, re.S):
        block = m.group(0)
        name_m = re.search(r'name\s+"?([\w.-]+)"?\s*;', block)
        to_m = re.search(r'\bto\s+"?([\w.-]+)"?\s*;', block)
        if name_m and to_m:
            name_of[to_m.group(1)] = name_m.group(1)
    return name_of


def build_line_to_from_map(base_text: str) -> dict:
    """triplex_line 'to' -> 'from', to walk backward from a service node to
    the transformer's secondary node (see wire_9500_helics.py for why this
    extra hop exists on this feeder)."""
    to_from = {}
    for m in re.finditer(r"object triplex_line \{.*?\n\}", base_text, re.S):
        block = m.group(0)
        f_m = re.search(r'\bfrom\s+"?([\w.-]+)"?\s*;', block)
        t_m = re.search(r'\bto\s+"?([\w.-]+)"?\s*;', block)
        if f_m and t_m:
            to_from[t_m.group(1)] = f_m.group(1)
    return to_from


def build_recorder_block(transformer_name: str) -> str:
    return (
        "object recorder {\n"
        f'    parent "{transformer_name}";\n'
        f"    file {RESULTS_DIR_REL}/{transformer_name}.csv;\n"
        "    interval 60;\n"
        "    property power_in,power_out;\n"
        "}"
    )


def main():
    base_text = BASE_GLM.read_text()
    transformer_name_of = build_transformer_name_map(base_text)
    line_to_from = build_line_to_from_map(base_text)

    with open(CLUSTER_MANIFEST) as f:
        clusters = json.load(f)
    with open(REMAINING_MANIFEST) as f:
        remaining = json.load(f)

    transformer_names = []

    for c in clusters:
        transformer_names.append(transformer_name_of[c["transformer_to_node"]])

    for r in remaining:
        transformer_to_node = line_to_from[r["parent_node"]]
        transformer_names.append(transformer_name_of[transformer_to_node])

    assert len(transformer_names) == len(set(transformer_names)) == 268, \
        f"expected 268 unique transformers, got {len(transformer_names)} ({len(set(transformer_names))} unique)"

    recorder_blocks = "\n".join(build_recorder_block(name) for name in transformer_names)
    new_base_text = base_text.rstrip("\n") + "\n\n" + recorder_blocks + "\n"
    BASE_GLM.write_text(new_base_text)

    print(f"Added {len(transformer_names)} transformer recorders to {BASE_GLM}")


if __name__ == "__main__":
    main()
