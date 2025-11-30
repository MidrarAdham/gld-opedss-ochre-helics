import os
from load_profiles import LoadProfiles

dataset_dir = f"{os.getcwd()}/datasets/cosimulation/"

lp = LoadProfiles(
    dataset_dir=dataset_dir,
    n_buildings=50,
    upgrades=['up00']
)


lp.run()

# Get a list of customer IDs:
customer_id = lp.load_profiles[0]

# Get a dataframe for a single customer
df = lp.customer_data[customer_id]

# Get a summary for customers
customer_summary = lp.customer_summaries[customer_id]


agg_all = lp.aggregate_customers_load_calculations ()


# self.aggregate_customers_load_calculations(transformer_kva=[15])