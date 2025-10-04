
"""
This script is for debugging data merging issues, specifically to diagnose
why observations are being dropped during the mapping of individual data
to the state space in the DataLoader.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.config.model_config import ModelConfig

def debug_data_merge():
    print("--- Starting Data Merge Debugging Script ---")

    config = ModelConfig()
    data_loader = DataLoader(config)

    # 1. Load preprocessed individual data and regional data
    # Ensure we load the latest preprocessed individual data
    config.individual_data_path = os.path.join(config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
    df_individual_raw = data_loader.load_individual_data()
    
    config.regional_data_path = os.path.join(config.processed_data_dir, 'geo_amenities.csv')
    df_region = data_loader.load_regional_data()

    print(f"Loaded df_individual_raw shape: {df_individual_raw.shape}")
    print(f"Loaded df_region shape: {df_region.shape}")

    # 2. Re-create state space (as done in DataLoader)
    ages = np.arange(config.age_min, config.age_max + 1)
    locations = sorted(df_region['provcd'].unique())
    
    state_space = pd.DataFrame(
        [(age, loc) for age in ages for loc in locations],
        columns=['age', 'prev_provcd']
    )
    state_space['state_index'] = state_space.index
    print(f"Re-created state space with {len(state_space)} states.")
    print(f"State space ages range: [{state_space['age'].min()}, {state_space['age'].max()}]")
    print(f"State space locations count: {len(state_space['prev_provcd'].unique())}")

    # 3. Simulate the merge and identify unmatched observations
    # Drop _merge column if it exists from previous runs
    if '_merge' in df_individual_raw.columns:
        df_individual_raw.drop(columns=['_merge'], inplace=True)

    df_merged_debug = pd.merge(
        df_individual_raw,
        state_space,
        left_on=['age_t', 'prev_provcd'],
        right_on=['age', 'prev_provcd'],
        how='left',
        indicator=True
    )

    unmatched_count = df_merged_debug['_merge'].value_counts().get('left_only', 0)
    print(f"\nTotal observations in df_individual_raw: {len(df_individual_raw)}")
    print(f"Merge indicator distribution:\n{df_merged_debug['_merge'].value_counts()}")
    print(f"Number of observations that failed to match state space: {unmatched_count}")

    if unmatched_count > 0:
        print("\n--- Analyzing Unmatched Observations ---")
        unmatched_df = df_merged_debug[df_merged_debug['_merge'] == 'left_only'].copy()
        
        print("Unique (age_t, prev_provcd) combinations in unmatched data:")
        print(unmatched_df[['age_t', 'prev_provcd']].drop_duplicates().sort_values(by=['age_t', 'prev_provcd']))

        print("\nStatistics for 'age_t' in unmatched data:")
        print(unmatched_df['age_t'].describe())
        print(f"Age_t values outside config range [{config.age_min}, {config.age_max}]:")
        out_of_range_ages = unmatched_df['age_t'][
            (unmatched_df['age_t'] < config.age_min) | (unmatched_df['age_t'] > config.age_max)
        ].unique()
        print(out_of_range_ages)

        print("\nStatistics for 'prev_provcd' in unmatched data:")
        print(unmatched_df['prev_provcd'].describe())
        print(f"Prev_provcd values not in df_region unique provcds:")
        region_provcds = set(df_region['provcd'].unique())
        unmatched_provcds = [p for p in unmatched_df['prev_provcd'].unique() if p not in region_provcds]
        print(unmatched_provcds)

    print("--- Data Merge Debugging Script Finished ---")

if __name__ == '__main__':
    debug_data_merge()
