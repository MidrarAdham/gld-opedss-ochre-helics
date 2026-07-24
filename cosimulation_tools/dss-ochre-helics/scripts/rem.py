import os
import pandas as pd

# Set your directory path
directory = "./profiles/one_week_wh_data/"  # <-- Change this

# Value to replace zero with
replacement_value = 4500

# Process each CSV file
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)

        # Load the CSV
        df = pd.read_csv(file_path)

        # Check the second column name (index 1)
        second_col = df.columns[1]

        # Update the first five rows of the second column if value is 0
        for i in range(min(5, len(df))):
            if df.at[i, second_col] == 0:
                df.at[i, second_col] = replacement_value

        # Save the updated CSV (overwriting the original)
        df.to_csv(file_path, index=False)

print("All applicable files updated.")
