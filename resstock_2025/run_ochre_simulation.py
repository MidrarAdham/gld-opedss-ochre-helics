import os
from ochre.cli import create_dwelling
from load_profiles import LoadProfiles
from ochre.utils import default_input_path


upgrades = ['up00']
dataset_dir = f"{os.getcwd()}/datasets/cosimulation/"
lp = LoadProfiles (dataset_dir=dataset_dir, upgrades=upgrades, n_buildings=1e4)
default_weather_file_name = "USA_OR_Portland.Intl.AP.726980_TMY3.epw"
default_weather_file = os.path.join(default_input_path, "Weather", default_weather_file_name)
# Returns the input load profiles:
input_paths = lp.input_files_handler ()

for input_path in input_paths:
    dwelling = create_dwelling (
                input_path=input_path,
                start_year=2025,
                start_month=1,
                start_day=1,
                initialization_time=1,
                time_res= 1,
                duration= 30,
                weather_file_or_path=default_weather_file
            )
    dwelling.simulate()
