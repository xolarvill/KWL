
"""
This script trains the machine learning plugins (nuisance functions),
specifically the wage prediction model, as described in the paper.
It uses LightGBM with K-fold cross-fitting to generate out-of-sample
predictions, which avoids overfitting and ensures the independence
between the nuisance function estimation and the structural parameter estimation.
"""

import sys
import os
import pandas as pd
import numpy as np # Added for log/exp transformations
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_person import preprocess_individual_data
from src.config.model_config import ModelConfig

def train_wage_model(config: ModelConfig):
    """
    Trains a LightGBM model to predict wages and saves the model and predictions.
    """
    print("--- Starting ML Plugin Training: Wage Model ---")

    # 1. Load preprocessed individual data and regional data
    try:
        df_individual = preprocess_individual_data(config.individual_data_path)
        df_region = pd.read_csv(os.path.join(config.processed_data_dir, 'geo_amenities.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Merge regional features into individual data for training
    # Rename region columns to avoid conflicts and clarify they are for the current location
    region_cols_to_merge = [
        'provcd', 'year', '常住人口万', '人均可支配收入（元） ',
        '地区基本经济面', '房价（元每平方）', 'amenity_climate',
        'amenity_health', 'amenity_education', 'amenity_public_services'
    ]
    df_region_for_merge = df_region[region_cols_to_merge].copy()
    df_region_for_merge.rename(columns={
        'provcd': 'provcd_t',
        'year': 'year_t',
        '常住人口万': 'pop_t',
        '人均可支配收入（元） ': 'disp_income_t',
        '地区基本经济面': 'econ_base_t',
        '房价（元每平方）': 'housing_price_t',
    }, inplace=True)

    # Debugging: Check merge keys before merge
    print("\n--- Debugging Merge Keys ---")
    print("df_individual['provcd_t'] unique values:", df_individual['provcd_t'].unique())
    print("df_individual['provcd_t'] dtype:", df_individual['provcd_t'].dtype)
    print("df_individual['year_t'] unique values:", df_individual['year_t'].unique())
    print("df_individual['year_t'] dtype:", df_individual['year_t'].dtype)
    print("\ndf_region_for_merge['provcd_t'] unique values:", df_region_for_merge['provcd_t'].unique())
    print("df_region_for_merge['provcd_t'] dtype:", df_region_for_merge['provcd_t'].dtype)
    print("df_region_for_merge['year_t'] unique values:", df_region_for_merge['year_t'].unique())
    print("df_region_for_merge['year_t'] dtype:", df_region_for_merge['year_t'].dtype)

    # Ensure data types are consistent for merging
    df_individual['provcd_t'] = df_individual['provcd_t'].astype(int)
    df_individual['year_t'] = df_individual['year_t'].astype(int)
    df_region_for_merge['provcd_t'] = df_region_for_merge['provcd_t'].astype(int)
    df_region_for_merge['year_t'] = df_region_for_merge['year_t'].astype(int)
    print("Data types for merge keys standardized to int.")

    df_merged = pd.merge(
        df_individual,
        df_region_for_merge,
        on=['provcd_t', 'year_t'],
        how='left',
        indicator=True # Add indicator to check merge success
    )
    print(f"Merged individual and regional data for ML training. Shape: {df_merged.shape}")

    # Debugging: Check merge results
    print("\n--- Debugging Merge Results ---")
    print("Merge indicator distribution:", df_merged['_merge'].value_counts())
    print("Number of NaNs in 'pop_t' after merge:", df_merged['pop_t'].isnull().sum())
    print("Number of NaNs in 'amenity_health' after merge:", df_merged['amenity_health'].isnull().sum())
    print("-----------------------------")

    # 3. Feature Engineering & Selection
    TARGET = 'income'
    # Transform target variable: log(income + 1) to handle zeros and skewness
    df_merged['log_income'] = np.log(df_merged[TARGET] + 1)
    ML_TARGET = 'log_income'

    FEATURES = [
        'age_t',
        # 'education', # Removed as it's not in clds.csv
        'gender', 
        'provcd_t',
        'prev_provcd',
        'is_at_hukou', # New feature
        'pop_t', # New regional feature
        'disp_income_t', # New regional feature
        'econ_base_t', # New regional feature
        'housing_price_t', # New regional feature
        'amenity_climate', # New regional feature
        'amenity_health', # New regional feature
        'amenity_education', # New regional feature
        'amenity_public_services', # New regional feature
    ]
    # Add age_t squared
    df_merged['age_t_sq'] = df_merged['age_t']**2
    FEATURES.append('age_t_sq')

    # Check if necessary columns exist after merge and feature engineering
    for col in [ML_TARGET] + FEATURES:
        if col not in df_merged.columns:
            print(f"Error: Column '{col}' not found in the merged data. Skipping training.")
            # As a placeholder, create a dummy prediction column
            df_merged['wage_predicted'] = df_merged[TARGET].fillna(df_merged[TARGET].mean())
            output_path = os.path.join(config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
            df_merged.to_csv(output_path, index=False)
            print(f"Saved data with dummy wage predictions to {output_path}")
            return

    # Drop rows with missing target or features for training
    df_clean = df_merged.dropna(subset=[ML_TARGET] + FEATURES).copy()
    print(f"Training on {len(df_clean)} observations after dropping NaNs. Original shape: {df_merged.shape}")

    # 4. K-Fold Cross-fitting
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_log_predictions = pd.Series(index=df_clean.index, dtype=float)

    # LightGBM hyperparameters (basic tuning)
    lgbm_params = {
        'objective': 'regression_l1', # MAE objective, more robust to outliers
        'metric': 'rmse',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1, # Suppress verbose output
        'n_jobs': -1, # Use all available cores
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_clean)):
        print(f"--- Training Fold {fold+1}/{N_SPLITS} ---")
        
        X_train, X_val = df_clean.iloc[train_idx][FEATURES], df_clean.iloc[val_idx][FEATURES]
        y_train, y_val = df_clean.iloc[train_idx][ML_TARGET], df_clean.iloc[val_idx][ML_TARGET]

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(20, verbose=False)]) # Increased patience
        
        oof_log_predictions.iloc[val_idx] = model.predict(X_val)

    # Add out-of-fold log predictions to the clean dataframe
    df_clean['log_wage_predicted'] = oof_log_predictions
    # Inverse transform to get wage_predicted on original scale
    df_clean['wage_predicted'] = np.exp(df_clean['log_wage_predicted']) - 1
    # Ensure non-negative wages
    df_clean['wage_predicted'] = np.maximum(df_clean['wage_predicted'], 0)
    
    # 5. Train final model on all data and save it
    print("\n--- Training final model on all available data ---")
    final_model = lgb.LGBMRegressor(**lgbm_params)
    final_model.fit(df_clean[FEATURES], df_clean[ML_TARGET])

    os.makedirs(config.ml_models_dir, exist_ok=True)
    model_path = os.path.join(config.ml_models_dir, 'wage_predictor.pkl')
    joblib.dump(final_model, model_path)
    print(f"Final wage model saved to: {model_path}")

    # 6. Save the data with OOF predictions
    # Merge predictions back into the original dataframe (df_merged) to keep all individuals
    df_merged = df_merged.merge(
        df_clean[['individual_id', 'year_t', 'log_wage_predicted', 'wage_predicted']],
        on=['individual_id', 'year_t'],
        how='left',
        suffixes=('_original', '')
    )
    
    # Fill missing predictions (e.g., for rows dropped due to NaNs in features) with mean
    df_merged['wage_predicted'].fillna(df_merged['wage_predicted'].mean(), inplace=True)
    df_merged['log_wage_predicted'].fillna(df_merged['log_wage_predicted'].mean(), inplace=True)

    output_path = os.path.join(config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
    df_merged.to_csv(output_path, index=False)
    print(f"Preprocessed data with improved out-of-sample wage predictions saved to: {output_path}")


if __name__ == '__main__':
    model_config = ModelConfig()
    train_wage_model(model_config)
