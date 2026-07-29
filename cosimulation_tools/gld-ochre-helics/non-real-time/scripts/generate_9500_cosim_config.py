'''
Author: Midrar Adham
Created: Tue Jul 28 2026

Generates the remaining HELICS/cosim config JSON for the IEEE 9500-node feeder,
mirroring cosim_master.py's role for powerflow_4node.glm. Run this after
wire_9500_helics.py has already renamed the GLM's ochre_house_load_<id> objects
and written config/9500/load_paths.json.
'''
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cosim_master import (
    create_ochre_federate_config,
    create_gridlabd_federate_config,
    create_ochre_helics_config,
    create_gridlabd_helics_config,
    create_master_config,
)

non_real_time_dir = Path(__file__).parent.parent
config_dir = non_real_time_dir / "config" / "9500"

if __name__ == "__main__":
    ochre_federate_script = "scripts/ochre_cosim.py"
    gridlabd_model_file = "models/9500/model_startup_9500.glm"
    load_paths_file = config_dir / "load_paths.json"
    ochre_helics_config_file = "config/9500/ochre_helics_config.json"
    gridlabd_helics_config_file = "config/9500/gld_config.json"
    master_config_file = "config/9500/master_cosim_config.json"

    ochre_federate_name = "OCHRE_Federate_9500"
    gridlabd_federate_name = "GridLABD_Federate_9500"
    cosimulation_name = "ochre_gridlabd_9500_cosimulation"

    with open(load_paths_file, "r") as f:
        load_paths = json.load(f)
    # keys are "load_<id>" (see wire_9500_helics.py); strip the prefix to
    # recover the building IDs the HELICS configs key on.
    building_ids = [key[len("load_"):] for key in load_paths.keys()]

    create_ochre_helics_config(
        ochre_federate_name=ochre_federate_name,
        building_ids=building_ids,
        output_file=ochre_helics_config_file,
    )

    create_gridlabd_helics_config(
        gridlabd_federate_name=gridlabd_federate_name,
        building_ids=building_ids,
        output_file=gridlabd_helics_config_file,
    )

    ochre_federate = create_ochre_federate_config(
        ochre_federate_script=ochre_federate_script,
        ochre_federate_name=ochre_federate_name,
    )
    # ochre_cosim.py reads config/9500/*.json instead of config/4node/*.json
    # when invoked with a "9500" argument (see ochre_cosim.py)
    ochre_federate["exec"] += " 9500"

    gridlabd_federate = create_gridlabd_federate_config(
        gridlabd_model_file=gridlabd_model_file,
        gridlabd_federate_name=gridlabd_federate_name,
    )

    create_master_config(
        cosimulation_name=cosimulation_name,
        federates=[ochre_federate, gridlabd_federate],
        output_file=master_config_file,
    )

    print(f"{len(building_ids)} building IDs")
    print(f"Wrote {ochre_helics_config_file}")
    print(f"Wrote {gridlabd_helics_config_file}")
    print(f"Wrote {master_config_file}")
