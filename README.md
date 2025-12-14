# ResStock 2025:

Clone this directory, and go to the dataset/ folder to see intructions about how to download ResStock 2025

# Quick Progress Overview
So far, I resolved some HELICS issues. Here are the important ones:

- [error] broker responded with error: duplicate broker name detected

This error was caused by an existing operation of HELICS that was not terminated. This was resolved by:

    - ps aux | grep "helic"
    - Then kill that process:
        - ps kill [PID]

# How to run the GLD-Python example
Currently, there is one example of HELICS - GridLAB-D integration. Here is how to run it:

- Open three terminal windows.
- In the first one, run the broker:
```
helics_broker -f 2 --loglevel=warning
```
- In the second terminal window, run the GridLAB-D federate:
```
gridlabd -D HELICS_CONFIG_FILE=powerflow_4node_gld_config.json powerflow_4node.glm
```
- In the third terminal window, run the python federate:
```
python3 powerflow_test.py
```

## Progress

- The glm file is a four node feeder. The feeder incorporates inverter-battery object.

- The inverter-battery object is controlled by setting its ```P_Out``` and ```Q_Out``` properties.

- These values are published correctly by the python federate.

- However, the recorder object (attached to the inverter-battery object) is not recording these values.

- I tried adding a ```triplex_load``` object instead, which worked as expected.

- I posted a question in the GridLAB-D forum, see [here](https://sourceforge.net/p/gridlab-d/discussion/842562/thread/e5a93df6c6/).

# Poetry usage:
This project uses poetry to manage the dependencies. In this section, a step-by-step guide showing how to install poetry, add the dependencies, and solve anticipated errors.

## Requirements:

### Python Requirements:

This project is built for:
```
Python 3.12.x
```

This constraint is enforced because OCHRE supports <3.13. In pyproject.toml:
```
requires-python = ">=3.12,<3.13"
```

## Installing Poetry (Ubuntu):

- Install Poetry:
```
curl -sSL https://install.python-poetry.org | python3 -
```
- Ensure it is on the Path:
```
export PATH="$HOME/.local/bin:$PATH"
```
- Verify it is installed correctly:
```
poetry --version
```

## Install Dependencies:
```
poetry install 
```
Here are the list dependencies that should be installed:

- numpy (>=1.26,<2.0)
- scipy
- pandas
- matplotlib
- seaborn
- click
- helics[cli]
- ochre-nrel