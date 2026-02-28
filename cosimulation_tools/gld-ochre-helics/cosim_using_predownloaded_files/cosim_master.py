import os
import sys
import json
from helics.cli import run


def create_load_paths_file(building_ids: list, resstock_data_dir: str, load_paths_file: str):
    """
    Creates a JSON file mapping each load to its ochre.csv path.
    This lets the OCHRE federate read as many load profiles as needed
    from a single file instead of passing each path on the command line.
    """
    load_paths = {}
    for idx, bldg_id in enumerate(building_ids):
        load_paths[f"load_{idx}"] = f"{resstock_data_dir}/{bldg_id}/up00/ochre.csv"

    with open(load_paths_file, "w") as f:
        json.dump(load_paths, f, indent=4)

    return load_paths


def create_ochre_federate_config(ochre_federate_script: str, ochre_federate_name: str):
    """
    Builds the HELICS federate config for OCHRE.
    The 'exec' field is the command HELICS will run in a separate process.
    """
    ochre_cmd = f"{sys.executable} -u {os.path.abspath(ochre_federate_script)}"
    ochre_federate = {
        "name": ochre_federate_name,
        "host": "localhost",
        "directory": ".",
        "exec": ochre_cmd,
    }

    return ochre_federate


def create_gridlabd_federate_config(gridlabd_model_file: str, gridlabd_federate_name: str):
    """
    Builds the HELICS federate config for GridLAB-D.
    """
    gridlabd_cmd = f"gridlabd {os.path.abspath(gridlabd_model_file)}"
    gridlabd_federate = {
        "name": gridlabd_federate_name,
        "host": "localhost",
        "directory": ".",
        "exec": gridlabd_cmd,
    }

    return gridlabd_federate


def create_ochre_helics_config(ochre_federate_name: str, building_ids: list, output_file: str):
    """
    Creates the HELICS config for the OCHRE federate.
    Generates one publication per building — these are the power values
    that OCHRE sends out for GridLAB-D to receive.
    """
    publications = []
    for idx in range(len(building_ids)):
        publications.append({
            "key": f"ochre_house_load_{idx}.constant_power_12",
            "type": "complex",
            "global": True,
        })

    config = {
        "name": ochre_federate_name,
        "coreType": "zmq",
        "coreInitString": "--federates=1",
        "period": 60.0,
        "uninterruptible": True,
        "terminate_on_error": True,
        "publications": publications,
        "subscriptions": [],
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)


def create_gridlabd_helics_config(gridlabd_federate_name: str, building_ids: list, output_file: str):
    """
    Creates the HELICS config for the GridLAB-D federate.
    Generates one subscription per building — these match the publications
    from OCHRE so GridLAB-D receives each building's power demand.
    """
    subscriptions = []
    for idx in range(len(building_ids)):
        subscriptions.append({
            "key": f"ochre_house_load_{idx}.constant_power_12",
            "type": "complex",
            "info": json.dumps({
                "object": f"ochre_house_load_{idx}",
                "property": "constant_power_12",
            }),
            "required": True,
        })

    config = {
        "name": gridlabd_federate_name,
        "coreInitString": "--federates=1 --loglevel=warning",
        "coreType": "zmq",
        "period": 60.0,
        "uninterruptible": False,
        "terminate_on_error": True,
        "subscriptions": subscriptions,
        "publications": [],
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)


def create_master_config(cosimulation_name: str, federates: list, output_file: str):
    """
    Creates the master HELICS config JSON that ties all federates together.
    Setting broker=True tells HELICS to start a broker automatically.
    """
    config = {
        "name": cosimulation_name,
        "broker": True,
        "federates": federates,
    }

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":

    # File paths
    ochre_federate_script = "scripts/ochre_cosim.py"
    gridlabd_model_file = "models/powerflow_4node.glm"
    load_paths_file = "config/load_paths.json"
    ochre_helics_config_file = "config/ochre_helics_config.json"
    gridlabd_helics_config_file = "config/powerflow_4node_gld_config.json"
    master_config_file = "config/master_cosim_config.json"

    # Names that are used by HELICS to identify each piece
    ochre_federate_name = "OCHRE_Federate"
    gridlabd_federate_name = "GridLABD_Federate"
    cosimulation_name = "ochre_gridlabd_cosimulation"

    # Dataset location which already exists at root folder of this project
    resstock_data_dir = "/home/deras/gld-opedss-ochre-helics/datasets/resstock_2025/load_profiles/tmp"

    # Building IDs to simulate - this is what is handled by one OCHRE federate
    building_ids = [385977, 292272, 32387, 188195, 504038, 370750, 258272, 193969]

    # Step 1: Create the load paths JSON file
    create_load_paths_file(
        building_ids=building_ids,
        resstock_data_dir=resstock_data_dir,
        load_paths_file=load_paths_file,
    )

    # Step 2: Create the OCHRE HELICS config (one publication per building)
    create_ochre_helics_config(
        ochre_federate_name=ochre_federate_name,
        building_ids=building_ids,
        output_file=ochre_helics_config_file,
    )

    # Step 3: Create the GridLAB-D HELICS config (one subscription per building, but one federate)
    create_gridlabd_helics_config(
        gridlabd_federate_name=gridlabd_federate_name,
        building_ids=building_ids,
        output_file=gridlabd_helics_config_file,
    )

    # Step 4: Build each federate's launch config
    ochre_federate = create_ochre_federate_config(
        ochre_federate_script=ochre_federate_script,
        ochre_federate_name=ochre_federate_name
    )

    gridlabd_federate = create_gridlabd_federate_config(
        gridlabd_model_file=gridlabd_model_file,
        gridlabd_federate_name=gridlabd_federate_name,
    )

    # Step 5: Create the master config and save it
    create_master_config(
        cosimulation_name=cosimulation_name,
        federates=[ochre_federate, gridlabd_federate],
        output_file=master_config_file,
    )

    # Step 6: Run the co-simulation (uncomment when ready)
    run(["--path", master_config_file])