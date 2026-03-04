import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
import os
import argparse
import sys
from datetime import datetime

DEFAULT_DISEASE = 'all'

def main():
    parser = argparse.ArgumentParser(description='Run Baseline Cox and Weibull Models')
    parser.add_argument('--disease', type=str, default=DEFAULT_DISEASE, help='Disease to analyze (default: all)')
    parser.add_argument('--feature_set', type=str, default='demographics', choices=['clinical', 'demographics'], help='Feature set to use (default: demographics)')
    parser.add_argument('--stratified', action='store_true', help='Run with stratification (default is unstratified)')
    parser.add_argument('--model', type=str, default='both', choices=['cox', 'weibull', 'both'], help='Model type to run (default: both)')
    parser.add_argument('--dry-run', action='store_true', help='Run quickly to verify pipeline')
    args = parser.parse_args()

    FEATURE_SET = args.feature_set
    UNSTRATIFIED = not args.stratified
    MODEL_TYPE = args.model
    
    if args.disease == 'all':
        diseases = ['hypertension', 'diabetes', 'dyslipidemia', 'obesity']
    else:
        diseases = [args.disease]
        
    for target_disease in diseases:
        run_baseline_for_disease(target_disease, FEATURE_SET, UNSTRATIFIED, MODEL_TYPE, args.dry_run)

def run_baseline_for_disease(DISEASE, FEATURE_SET, UNSTRATIFIED, MODEL_TYPE, dry_run=False):
    DATA_DIR = f'tfrecords/{DISEASE}_triplet'
    
    if not os.path.exists(DATA_DIR):
        print(f"\nSkipping {DISEASE}: Data directory {DATA_DIR} does not exist.")
        return
    

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    strat_suffix = "_unstratified" if UNSTRATIFIED else ""
    out_dir = f'baseline_{DISEASE}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting {DISEASE} Baseline Analysis ({FEATURE_SET}, Model: {MODEL_TYPE}).")
    print(f"Output directory: {out_dir}")

    csv_path = f'{DATA_DIR}/metadata.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Metadata file not found: {csv_path}")
        sys.exit(1)
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    try:
        if df['height1'].mean() > 3:
            df['bmi'] = df['weight1'] / ((df['height1'] / 100) ** 2)
        else:
            df['bmi'] = df['weight1'] / (df['height1'] ** 2)
    except:
        df['bmi'] = np.nan
        
    df['dt1'] = pd.to_datetime(df['dt1'], errors='coerce')
    df['dt2'] = pd.to_datetime(df['dt2'], errors='coerce')
    

    df['duration1_2'] = (df['dt2'] - df['dt1']).dt.days / 365.25
        
    target_event_col = f'{DISEASE}2'
    
    if target_event_col not in df.columns:
        print(f"Error: Target column {target_event_col} not found in metadata.")
        sys.exit(1)
        
    df['disease2'] = df[target_event_col]
    
    target_cols = ['disease2', 'duration1_2']
    meta_cols = ['ID', 'shard']
    
    df['sex_code'] = df['sex'].apply(lambda x: 1 if str(x).upper().startswith('M') else 0)
    
    if FEATURE_SET == 'clinical':
        final_features = ['age1', 'sex_code', 'bmi', 'sbp1', 'dbp1']
    else:
        final_features = ['age1', 'sex_code']
    
    model_df = df[final_features + target_cols + ['shard']].copy()
    
    model_df = model_df.rename(columns={'sex_code': 'sex'})
    final_features = [f if f != 'sex_code' else 'sex' for f in final_features]

    initial_len = len(model_df)

    model_df = model_df.dropna(subset=target_cols)
    
    missing_counts = model_df[final_features].isnull().sum()
    print("Missing values per feature:\n", missing_counts)
   
    model_df['disease2'] = model_df['disease2'].astype(float).astype(int)
    model_df['duration1_2'] = model_df['duration1_2'].astype(float)
    
    epsilon = 1e-5
    model_df['duration1_2'] = model_df['duration1_2'].apply(lambda x: max(x, epsilon))

    print(f"Data prepared: {len(model_df)} samples (dropped {initial_len - len(model_df)} due to missing targets)")

    summary_results_cox = []
    summary_results_weibull = []
    
    for fold in range(5):
        print(f"\n--- Fold {fold} ---")
        
        n_shards = 10
        test_shards = [2*fold, 2*fold+1]
        train_shards = [i for i in range(n_shards) if i not in test_shards]
        
        if dry_run:
            train_df = model_df[model_df['shard'].isin(train_shards)].sample(100)
            test_df = model_df[model_df['shard'].isin(test_shards)].sample(20)
        else:
            train_df = model_df[model_df['shard'].isin(train_shards)].copy()
            test_df = model_df[model_df['shard'].isin(test_shards)].copy()
        
        fill_values = train_df[final_features].mean()
        train_df[final_features] = train_df[final_features].fillna(fill_values)
        test_df[final_features] = test_df[final_features].fillna(fill_values)

        train_df['age_group'] = (train_df['age1'] // 10).astype(int)
        test_df['age_group'] = (test_df['age1'] // 10).astype(int)
        
        fit_cols = final_features + target_cols
        strata_cols = None
        
        if not UNSTRATIFIED:
            fit_cols = final_features + ['age_group'] + target_cols
            strata_cols = ['sex', 'age_group']
        else:
            pass

        if MODEL_TYPE in ['cox', 'both']:
            cph = CoxPHFitter()
            try:
                cph.fit(train_df[fit_cols], 
                        duration_col='duration1_2', 
                        event_col='disease2', 
                        strata=strata_cols,
                        robust=True)
                
                c_index = cph.score(test_df, scoring_method="concordance_index")
                print(f"Fold {fold} Cox C-index: {c_index:.4f}")
                
                for cov in cph.summary.index:
                    summary_row = cph.summary.loc[cov]
                    summary_results_cox.append({
                        'Fold': fold,
                        'Model': 'Baseline',
                        'Covariate': cov,
                        'Strata': ', '.join(strata_cols) if strata_cols else 'None',
                        'HR': np.exp(summary_row['coef']),
                        'HR_Lower_95': summary_row['exp(coef) lower 95%'],
                        'HR_Upper_95': summary_row['exp(coef) upper 95%'],
                        'P-value': summary_row['p'],
                        'Concordance': c_index,
                        'Train_Size': len(train_df),
                        'Test_Size': len(test_df)
                    })
                
                pass
            except Exception as e:
                print(f"Fold {fold} Cox failed: {e}")

        if MODEL_TYPE in ['weibull', 'both']:
            aft = WeibullAFTFitter()
            try:
                weibull_fit_cols = final_features + target_cols
                if not UNSTRATIFIED:
                    pass 

                aft.fit(train_df[weibull_fit_cols], 
                        duration_col='duration1_2', 
                        event_col='disease2',
                        robust=True)
                
                c_index_w = aft.score(test_df, scoring_method="concordance_index")
                print(f"Fold {fold} Weibull C-index: {c_index_w:.4f}")

                if 'lambda_' in aft.summary.index.get_level_values(0):
                    for cov in aft.summary.loc['lambda_'].index:
                        summary_row = aft.summary.loc[('lambda_', cov)]
                        summary_results_weibull.append({
                            'Fold': fold,
                            'Model': 'Baseline',
                            'Covariate': cov,
                            'Strata': 'None',
                            'Coef': summary_row['coef'],
                            'TimeRatio': np.exp(summary_row['coef']),
                            'TR_Lower_95': summary_row['exp(coef) lower 95%'],
                            'TR_Upper_95': summary_row['exp(coef) upper 95%'],
                            'P-value': summary_row['p'],
                            'Concordance': c_index_w,
                            'Train_Size': len(train_df),
                            'Test_Size': len(test_df)
                        })
                
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Fold {fold} Weibull failed: {e}")
            
    print("\n--- Overall (All Data) ---")
    overall_df = model_df.copy()
    if dry_run:
        overall_df = overall_df.sample(min(120, len(overall_df)))
        
    fill_values_overall = overall_df[final_features].mean()
    overall_df[final_features] = overall_df[final_features].fillna(fill_values_overall)
    overall_df['age_group'] = (overall_df['age1'] // 10).astype(int)
    
    fit_cols_overall = final_features + target_cols
    strata_cols_overall = None
    if not UNSTRATIFIED:
        fit_cols_overall = final_features + ['age_group'] + target_cols
        strata_cols_overall = ['sex', 'age_group']

    if MODEL_TYPE in ['cox', 'both']:
        try:
            cph_overall = CoxPHFitter()
            cph_overall.fit(overall_df[fit_cols_overall], 
                    duration_col='duration1_2', 
                    event_col='disease2', 
                    strata=strata_cols_overall,
                    robust=True)
            
            c_index_overall = cph_overall.score(overall_df, scoring_method="concordance_index")
            print(f"Overall Cox C-index: {c_index_overall:.4f}")
            
            for cov in cph_overall.summary.index:
                summary_row_occ = cph_overall.summary.loc[cov]
                summary_results_cox.append({
                    'Fold': 'Pooled',
                    'Model': 'Baseline',
                    'Covariate': cov,
                    'Strata': ', '.join(strata_cols_overall) if strata_cols_overall else 'None',
                    'HR': np.exp(summary_row_occ['coef']),
                    'HR_Lower_95': summary_row_occ['exp(coef) lower 95%'],
                    'HR_Upper_95': summary_row_occ['exp(coef) upper 95%'],
                    'P-value': summary_row_occ['p'],
                    'Concordance': c_index_overall,
                    'Train_Size': len(overall_df),
                    'Test_Size': len(overall_df)
                })

                pass
        except Exception as e:
            print(f"Overall Cox failed: {e}")

    if MODEL_TYPE in ['weibull', 'both']:
        try:
            aft_overall = WeibullAFTFitter()
            weibull_fit_cols_overall = final_features + target_cols
            aft_overall.fit(overall_df[weibull_fit_cols_overall], 
                    duration_col='duration1_2', 
                    event_col='disease2',
                    robust=True)
            
            c_index_w_overall = aft_overall.score(overall_df, scoring_method="concordance_index")
            print(f"Overall Weibull C-index: {c_index_w_overall:.4f}")
            
            if 'lambda_' in aft_overall.summary.index.get_level_values(0):
                for cov in aft_overall.summary.loc['lambda_'].index:
                    summary_row_owc = aft_overall.summary.loc[('lambda_', cov)]
                    summary_results_weibull.append({
                        'Fold': 'Pooled',
                        'Model': 'Baseline',
                        'Covariate': cov,
                        'Strata': 'None',
                        'Coef': summary_row_owc['coef'],
                        'TimeRatio': np.exp(summary_row_owc['coef']),
                        'TR_Lower_95': summary_row_owc['exp(coef) lower 95%'],
                        'TR_Upper_95': summary_row_owc['exp(coef) upper 95%'],
                        'P-value': summary_row_owc['p'],
                        'Concordance': c_index_w_overall,
                        'Train_Size': len(overall_df),
                        'Test_Size': len(overall_df)
                    })
            
                pass
        except Exception as e:
            print(f"Overall Weibull failed: {e}")
            
    print("\n--- Summary Results ---")
    
    if summary_results_cox:
        cox_df = pd.DataFrame(summary_results_cox)
        print("Cox Baseline Results Preview:")
        print(cox_df.head())
        cox_df.to_csv(os.path.join(out_dir, 'cox_baseline_cv_summary.csv'), index=False)
        
        cox_folds = cox_df[cox_df['Fold'] != 'Pooled']
        mean_c_index_cox = cox_folds.groupby('Fold')['Concordance'].first().mean() if not cox_folds.empty else np.nan
        std_c_index_cox = cox_folds.groupby('Fold')['Concordance'].first().std() if not cox_folds.empty else np.nan
    
    if summary_results_weibull:
        weibull_df = pd.DataFrame(summary_results_weibull)
        print("Weibull Baseline Results Preview:")
        print(weibull_df.head())
        weibull_df.to_csv(os.path.join(out_dir, 'weibull_baseline_cv_summary.csv'), index=False)
        
        weibull_folds = weibull_df[weibull_df['Fold'] != 'Pooled']
        mean_c_index_weibull = weibull_folds.groupby('Fold')['Concordance'].first().mean() if not weibull_folds.empty else np.nan
        std_c_index_weibull = weibull_folds.groupby('Fold')['Concordance'].first().std() if not weibull_folds.empty else np.nan

if __name__ == "__main__":
    main()
