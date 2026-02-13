import os
import numpy as np
import pandas as pd
import seaborn as sns
from config import load_config
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

cfg = load_config ()

metadata_dir = "/home/deras/gld-opedss-ochre-helics/datasets/resstock_2025/scripts/OR_upgrade0.csv"

validated_dataset_dir = cfg['data']['dataset_dir']
validated_bldg_ids = [int(bldg_id) for bldg_id in os.listdir (validated_dataset_dir)]


df_metadata = pd.read_csv (metadata_dir, low_memory=False)

df_metadata = df_metadata [df_metadata['bldg_id'].isin (validated_bldg_ids)].reset_index ()

keep_cols = ['bldg_id','in.electric_panel_service_rating..a', 'in.electric_panel_service_rating_bin..a',
             'out.params.panel_load_total_load.2023_nec_existing_dwelling_load_based..w',
             'out.params.panel_load_occupied_capacity.2023_nec_existing_dwelling_load_based..a',
             'out.params.panel_load_headroom_capacity.2023_nec_existing_dwelling_load_based..a',
             'in.electric_panel_breaker_space_total_count', 'out.params.panel_breaker_space_occupied_count',
             'out.params.panel_breaker_space_headroom_count'
             ]

df_metadata_filtered = df_metadata [keep_cols]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

df_sorted = df_metadata_filtered.sort_values('in.electric_panel_service_rating..a')

# Add these 4 lines:
unique_bins = df_sorted['in.electric_panel_service_rating_bin..a'].unique()

palette = sns.color_palette("flare", n_colors=len(unique_bins))

color_map = dict(zip(unique_bins, palette))

colors = df_sorted['in.electric_panel_service_rating_bin..a'].map(color_map)

# Find the shitty buildings:

bldg_409590_position = df_sorted[df_sorted['bldg_id'] == 409590].index[0]
bldg_409590_position_in_plot = df_sorted.index.get_loc(bldg_409590_position)


bldg_355669_position = df_sorted[df_sorted['bldg_id'] == 355669].index[0]
bldg_355669_position_in_plot = df_sorted.index.get_loc(bldg_355669_position)


axes[0].barh(
    y=range(len(df_sorted)),
    width=df_sorted['in.electric_panel_service_rating..a'],
    color=colors  # Changed from 'orange' to colors
)

axes[0].barh(
    y=bldg_409590_position_in_plot,
    width=df_sorted.loc[bldg_409590_position, 'in.electric_panel_service_rating..a'],
    color='gray', edgecolor= 'black', linewidth=2
)

axes[0].barh(
    y=bldg_355669_position_in_plot,
    width=df_sorted.loc[bldg_355669_position, 'in.electric_panel_service_rating..a'],
    color='slategray', edgecolor= 'black', linewidth=2
)

axes[0].set_yticks(range(len(df_sorted)))
axes[0].set_yticklabels(df_sorted['bldg_id'])
axes[0].set_xlabel('Service Rating (A)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Building ID', fontsize=13, fontweight='bold')
axes[0].set_title('Electric Panel Service Rating by Building', fontsize=16, fontweight='bold', pad=20)
axes[0].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
axes[0].set_axisbelow(True)

axes[1].barh(
    y=range(len(df_sorted)),
    width=df_sorted['out.params.panel_load_occupied_capacity.2023_nec_existing_dwelling_load_based..a'],
    color=colors
)

axes[1].barh(
    y=bldg_409590_position_in_plot,
    width=df_sorted.loc[bldg_409590_position,
                        'out.params.panel_load_occupied_capacity.2023_nec_existing_dwelling_load_based..a'],
                        color='gray', edgecolor='black', linewidth=2
                        )

axes[1].barh(
    y=bldg_355669_position_in_plot,
    width=df_sorted.loc[bldg_355669_position,
                        'out.params.panel_load_occupied_capacity.2023_nec_existing_dwelling_load_based..a'],
                        color='slategray', edgecolor='black', linewidth=2
                        )

axes[1].set_yticks(range(len(df_sorted)))
axes[1].set_yticklabels([])  # Don't repeat building IDs
axes[1].set_xlabel('Occupied Capacity (A)', fontsize=13, fontweight='bold')
axes[1].set_title('Panel Load Occupied Capacity', fontsize=14, fontweight='bold', pad=20)
axes[1].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
axes[1].set_axisbelow(True)

axes[2].barh(
    y=bldg_409590_position_in_plot,
    width=df_sorted.loc[bldg_409590_position,
                        'out.params.panel_load_headroom_capacity.2023_nec_existing_dwelling_load_based..a'],
                        color='gray', edgecolor='black', linewidth=2
                        )

axes[2].barh(
    y=bldg_355669_position_in_plot,
    width=df_sorted.loc[bldg_355669_position,
                        'out.params.panel_load_headroom_capacity.2023_nec_existing_dwelling_load_based..a'],
                        color='slategray', edgecolor='black', linewidth=2
                        )

axes[2].set_yticks(range(len(df_sorted)))
axes[2].set_yticklabels([])  # Don't repeat building IDs
axes[2].set_xlabel('Panel Load Headroom Rating (A)', fontsize=13, fontweight='bold')
axes[2].set_title('Headroom', fontsize=14, fontweight='bold', pad=20)
axes[2].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
axes[2].set_axisbelow(True)


legend_elements = [Patch(facecolor=color_map[bin_val], label=bin_val) 
                   for bin_val in unique_bins]

legend_elements.append(Patch(facecolor='gray', edgecolor='black', linewidth=2, 
                              label='Building 409590'))

legend_elements.append(Patch(facecolor='slategray', edgecolor='black', linewidth=2, 
                              label='Building 355669'))

axes[2].legend(handles=legend_elements, title='Rating Bin', loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig ('./panel_info_data_resstock.png', dpi = 150)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))



axes[0].barh(
    y=range(len(df_sorted)),
    width=df_sorted['in.electric_panel_breaker_space_total_count'],
    color=colors  # Changed from 'orange' to colors
)

axes[0].barh(
    y=bldg_409590_position_in_plot,
    width=df_sorted.loc[bldg_409590_position, 'in.electric_panel_breaker_space_total_count'],
    color='gray', edgecolor= 'black', linewidth=2
)

axes[0].barh(
    y=bldg_355669_position_in_plot,
    width=df_sorted.loc[bldg_355669_position, 'in.electric_panel_breaker_space_total_count'],
    color='slategray', edgecolor= 'black', linewidth=2
)

axes[0].set_yticks(range(len(df_sorted)))
axes[0].set_yticklabels(df_sorted['bldg_id'])
axes[0].set_xlabel('Total Electric Panel Count', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Building ID', fontsize=13, fontweight='bold')
axes[0].set_title('Total Electric Panel Circuit Breaker Slots', fontsize=16, fontweight='bold', pad=20)
axes[0].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
axes[0].set_axisbelow(True)

axes[1].barh(
    y=range(len(df_sorted)),
    width=df_sorted['out.params.panel_breaker_space_occupied_count'],
    color=colors
)

axes[1].barh(
    y=bldg_409590_position_in_plot,
    width=df_sorted.loc[bldg_409590_position,
                        'out.params.panel_breaker_space_occupied_count'],
                        color='gray', edgecolor='black', linewidth=2
                        )

axes[1].barh(
    y=bldg_355669_position_in_plot,
    width=df_sorted.loc[bldg_355669_position,
                        'out.params.panel_breaker_space_occupied_count'],
                        color='slategray', edgecolor='black', linewidth=2
                        )

axes[1].set_yticks(range(len(df_sorted)))
axes[1].set_yticklabels([])  # Don't repeat building IDs
axes[1].set_xlabel('Occupied Service Panel Count', fontsize=13, fontweight='bold')
axes[1].set_title('Electric Panel Occupied Breaker Slots', fontsize=14, fontweight='bold', pad=20)
axes[1].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
axes[1].set_axisbelow(True)

axes[2].barh(
    y=bldg_409590_position_in_plot,
    width=df_sorted.loc[bldg_409590_position,
                        'out.params.panel_breaker_space_headroom_count'],
                        color='gray', edgecolor='black', linewidth=2
                        )

axes[2].barh(
    y=bldg_355669_position_in_plot,
    width=df_sorted.loc[bldg_355669_position,
                        'out.params.panel_breaker_space_headroom_count'],
                        color='slategray', edgecolor='black', linewidth=2
                        )

axes[2].set_yticks(range(len(df_sorted)))
axes[2].set_yticklabels([])  # Don't repeat building IDs
axes[2].set_xlabel('Electric Panel Circuit Breaker Slots Available Count', fontsize=13, fontweight='bold')
axes[2].set_title('Headroom', fontsize=14, fontweight='bold', pad=20)
axes[2].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
axes[2].set_axisbelow(True)


legend_elements = [Patch(facecolor=color_map[bin_val], label=bin_val) 
                   for bin_val in unique_bins]

legend_elements.append(Patch(facecolor='gray', edgecolor='black', linewidth=2, 
                              label='Building 409590'))

legend_elements.append(Patch(facecolor='slategray', edgecolor='black', linewidth=2, 
                              label='Building 355669'))

axes[2].legend(handles=legend_elements, title='Rating Bin', loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig ('./panel_info_data_resstock_slots.png', dpi = 150)
plt.show()