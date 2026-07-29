import pandas as pd

# 1. Read NREL ResStock file (Skipping any initial blank/metadata headers if present)
df = pd.read_csv("portland_weather.csv")

# 2. Extract and format the columns to GridLAB-D standard naming conventions
# Map ResStock headers (e.g., 'Dry Bulb Temperature', 'Direct Normal Radiation') 
# directly into your required GridLAB-D column order.
gridlabd_df = pd.DataFrame()

# Re-generate clean 2025 timestamps matching your clock profile
gridlabd_df['timestamp'] = pd.date_range(start="2025-01-01 00:00:00", periods=8760, freq="h").strftime("%Y-%m-%d %H:%M:%S")

# Map standard variables (verify names in your downloaded CSV)
gridlabd_df['temperature'] = df['Dry Bulb Temperature']
gridlabd_df['humidity'] = df['Relative Humidity'] / 100.0  # GridLAB-D wants 0.0-1.0
gridlabd_df['wind_speed'] = df['Wind Speed']
gridlabd_df['solar_dir'] = df['Direct Normal Radiation']
gridlabd_df['solar_diff'] = df['Diffuse Horizontal Radiation']

# 3. Write out file with the required GridLAB-D header tag
with open("portland_weather_2025.csv", "w") as f:
    f.write("$weather_data\n")
    gridlabd_df.to_csv(f, index=False)

print("Conversion complete! Created portland_weather_2025.csv")
