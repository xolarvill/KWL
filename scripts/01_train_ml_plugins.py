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

    # 1. Load preprocessed individual data
    try:
        df_individual = preprocess_individual_data(config.individual_data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Feature Engineering & Selection
    # Define target and features
    TARGET = 'income' # Confirmed from clds.csv header
    # For the model to be meaningful, we need to control for selection.
    # A full implementation might involve a Heckman correction or more complex features.
    # For now, we use the available individual characteristics.
    FEATURES = [
        'age_t',
        'education', 
        'gender', 
        'provcd_t', # Current province might capture regional wage levels
        'prev_provcd' # Previous province might capture path dependency
    ]

    # Check if necessary columns exist
    for col in [TARGET] + ['age_t', 'education', 'gender', 'provcd_t', 'prev_provcd']:
        # Note: preprocess_individual_data renames some columns to *_t
        if col not in df_individual.columns:
            print(f"Error: Column '{col}' not found in the individual data. Skipping training.")
            # As a placeholder, create a dummy prediction column
            df_individual['wage_predicted'] = df_individual.get(TARGET, 10000)
            output_path = os.path.join(config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
            df_individual.to_csv(output_path, index=False)
            print(f"Saved data with dummy wage predictions to {output_path}")
            return

    # Drop rows with missing target or features
    df_clean = df_individual.dropna(subset=[TARGET] + FEATURES).copy()
    print(f"Training on {len(df_clean)} observations after dropping NaNs.")

    # 3. K-Fold Cross-fitting
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_predictions = pd.Series(index=df_clean.index, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_clean)):
        print(f"--- Training Fold {fold+1}/{N_SPLITS} ---")
        
        X_train, X_val = df_clean.iloc[train_idx][FEATURES], df_clean.iloc[val_idx][FEATURES]
        y_train, y_val = df_clean.iloc[train_idx][TARGET], df_clean.iloc[val_idx][TARGET]

        # Initialize and train the model
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        
        # Store out-of-fold predictions
        preds = model.predict(X_val)
        oof_predictions.iloc[val_idx] = preds

    # Add predictions to the dataframe
    df_clean['wage_predicted'] = oof_predictions
    
    # 4. Train final model on all data and save it
    print("\n--- Training final model on all available data ---")
    final_model = lgb.LGBMRegressor(random_state=42)
    final_model.fit(df_clean[FEATURES], df_clean[TARGET])

    # Create directory if it doesn't exist
    os.makedirs(config.ml_models_dir, exist_ok=True)
    model_path = os.path.join(config.ml_models_dir, 'wage_predictor.pkl')
    joblib.dump(final_model, model_path)
    print(f"Final wage model saved to: {model_path}")

    # 5. Save the data with OOF predictions
    # Merge predictions back into the original dataframe to keep all individuals
    df_individual['wage_predicted'] = df_clean['wage_predicted']
    # For individuals who had missing values, we can fill with the mean prediction
    df_individual['wage_predicted'].fillna(df_individual['wage_predicted'].mean(), inplace=True)

    output_path = os.path.join(config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
    df_individual.to_csv(output_path, index=False)
    print(f"Preprocessed data with out-of-sample wage predictions saved to: {output_path}")


if __name__ == '__main__':
    model_config = ModelConfig()
    train_wage_model(model_config)