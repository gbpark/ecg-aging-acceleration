import os
import glob
import pandas as pd
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter, WeibullFitter
from lifelines.utils import restricted_mean_survival_time
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

def evaluate_survival(df_results, fold_or_pooled, model_name, exp_dir):
    if fold_or_pooled == 'Pooled':
        out_dir = os.path.join(exp_dir, 'Pooled', model_name)
    else:
        out_dir = os.path.join(exp_dir, f'fold_{fold_or_pooled}', model_name)
    
    csvs_dir = os.path.join(out_dir, 'csvs')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(csvs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    res_full = None
    res_strat_sex = None
    res_raw_strat = None
    res_std_strat = None
    res_weibull = None
    
    try:
        df_results = df_results.copy()
        
        if 'residual' not in df_results.columns or 'age1' not in df_results.columns or 'sex' not in df_results.columns:
            print(f"    Missing required columns for {fold_or_pooled} model {model_name}. Skipping.")
            return None, None, None, None
            
        df_results['residual_std'] = (df_results['residual'] - df_results['residual'].mean()) / df_results['residual'].std()
        df_results['age_group'] = (df_results['age1'] // 10).astype(int)
        
        event_col = 'disease2' if 'disease2' in df_results.columns else 'event'
        dur_col = 'duration' if 'duration' in df_results.columns else 'duration1_2'
        
        df_results = df_results.rename(columns={event_col: 'event', dur_col: 'duration'})
        
        epsilon = 1e-5
        df_results['duration'] = df_results['duration'].apply(lambda x: max(x, epsilon))
        
        cph_full = CoxPHFitter(penalizer=0.01)
        cph_strat_sex = CoxPHFitter(penalizer=0.01)
        cph_raw_strat = CoxPHFitter(penalizer=0.01)
        cph_std_strat = CoxPHFitter(penalizer=0.01)

        # No strata, age1, sex, residual
        cph_full.fit(df_results[['duration', 'event', 'age1', 'sex', 'residual']], duration_col='duration', event_col='event', robust=True)
        summary_row_full = cph_full.summary.loc['residual']

        # Strata: sex, covariates: residual
        cph_strat_sex.fit(df_results[['duration', 'event', 'residual', 'sex']], duration_col='duration', event_col='event', strata=['sex'], robust=True)
        summary_row_strat_sex = cph_strat_sex.summary.loc['residual']

        # Strata: sex, age_group, covariates: residual
        df_cph = df_results[['duration', 'event', 'residual', 'sex', 'age_group']].copy()
        cph_raw_strat.fit(df_cph, duration_col='duration', event_col='event', strata=['sex', 'age_group'], robust=True)
        summary_row_raw_strat = cph_raw_strat.summary.loc['residual']

        # Strata: sex, age_group, covariates: residual_std
        cph_std_strat.fit(df_results[['duration', 'event', 'residual_std', 'sex', 'age_group']], duration_col='duration', event_col='event', strata=['sex', 'age_group'], robust=True)
        summary_row_std_strat = cph_std_strat.summary.loc['residual_std']

        # Schoenfeld Test
        try:
            res_sch = cph_raw_strat.compute_residuals(df_cph, kind='scaled_schoenfeld')
            if 'residual' in res_sch.columns:
                plt.figure(figsize=(10, 6))
                
                times = df_cph.loc[res_sch.index, 'duration'].values
                res_std = res_sch['residual'].values
                
                sort_idx = np.argsort(times)
                times = times[sort_idx]
                res_std = res_std[sort_idx]
                
                valid_idx = ~np.isnan(res_std) & ~np.isnan(times)
                if np.sum(valid_idx) > 1:
                    _, p_val = pearsonr(times[valid_idx], res_std[valid_idx])
                    p_val_str = f"p={p_val:.4f}"
                else:
                    p_val_str = "p=N/A"
                
                plt.scatter(times, res_std, alpha=0.3, s=15, color='gray')
                
                smoothed = lowess(res_std, times, frac=0.2)
                plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
                
                plt.axhline(0, color='black', linestyle='--', linewidth=1)
                
                plt.title(f'Schoenfeld Residuals - {model_name} {fold_or_pooled} ({p_val_str})')
                plt.xlabel('Years')
                plt.ylabel('Scaled Schoenfeld Residual')
                
                plt.xlim(left=0)
                max_duration = int(np.ceil(times.max()))
                plt.xticks(np.arange(0, max_duration + 1, 1))
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'schoenfeld_{fold_or_pooled}.png'))
                plt.close('all')
        except Exception as e:
            print(f"    Schoenfeld Plot Failed: {e}")
        
        res_full = {
            'Fold': fold_or_pooled,
            'Model': model_name,
            'Covariates': 'age1, sex, residual',
            'Strata': 'None',
            'HR': np.exp(summary_row_full['coef']),
            'HR_Lower_CI': summary_row_full['exp(coef) lower 95%'],
            'HR_Upper_CI': summary_row_full['exp(coef) upper 95%'],
            'P-value': summary_row_full['p'],
            'Concordance': cph_full.concordance_index_,
            'N': len(df_results)
        }

        res_strat_sex = {
            'Fold': fold_or_pooled,
            'Model': model_name,
            'Covariates': 'residual',
            'Strata': 'sex',
            'HR': np.exp(summary_row_strat_sex['coef']),
            'HR_Lower_CI': summary_row_strat_sex['exp(coef) lower 95%'],
            'HR_Upper_CI': summary_row_strat_sex['exp(coef) upper 95%'],
            'P-value': summary_row_strat_sex['p'],
            'N': len(df_results)
        }

        res_raw_strat = {
            'Fold': fold_or_pooled,
            'Model': model_name,
            'Covariates': 'residual',
            'Strata': 'sex, age_group',
            'HR': np.exp(summary_row_raw_strat['coef']),
            'HR_Lower_CI': summary_row_raw_strat['exp(coef) lower 95%'],
            'HR_Upper_CI': summary_row_raw_strat['exp(coef) upper 95%'],
            'P-value': summary_row_raw_strat['p'],
            'N': len(df_results)
        }
        
        res_std_strat = {
            'Fold': fold_or_pooled,
            'Model': model_name,
            'Covariates': 'residual_std',
            'Strata': 'sex, age_group',
            'HR': np.exp(summary_row_std_strat['coef']),
            'HR_Lower_CI': summary_row_std_strat['exp(coef) lower 95%'],
            'HR_Upper_CI': summary_row_std_strat['exp(coef) upper 95%'],
            'P-value': summary_row_std_strat['p'],
            'N': len(df_results)
        }
        
        # residual > 0 → Accelerated, residual <= 0 → Decelerated
        df_results['group'] = df_results['residual'].apply(lambda x: 'Accelerated' if x > 0 else 'Decelerated')

        t_max = df_results['duration'].max()
        max_duration = int(np.ceil(t_max))

        grp_dec = df_results[df_results['group'] == 'Decelerated']
        grp_acc = df_results[df_results['group'] == 'Accelerated']

        # KaplanMeier Plot
        km_rmst_vals = {} 

        plt.figure(figsize=(10, 6))

        if len(grp_dec) > 0:
            kmf_dec = KaplanMeierFitter()
            kmf_dec.fit(grp_dec["duration"], grp_dec["event"], label=f"Decelerated (n={len(grp_dec)})")
            km_rmst = restricted_mean_survival_time(kmf_dec, t=t_max)
            km_rmst_vals['Decelerated'] = km_rmst
            kmf_dec.plot_survival_function(ci_show=True)
        if len(grp_acc) > 0:
            kmf_acc = KaplanMeierFitter()
            kmf_acc.fit(grp_acc["duration"], grp_acc["event"], label=f"Accelerated (n={len(grp_acc)})")
            km_rmst = restricted_mean_survival_time(kmf_acc, t=t_max)
            km_rmst_vals['Accelerated'] = km_rmst
            kmf_acc.plot_survival_function(ci_show=True)

        plt.title(f'Kaplan-Meier Survival - {model_name} {fold_or_pooled}')
        plt.xlabel('Years')
        plt.ylabel('Survival Probability')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, max_duration + 1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'km_plot_{fold_or_pooled}.png'))
        plt.close('all')

        # Weibull Plot
        aft = WeibullAFTFitter(penalizer=0.01)
        aft.fit(df_results[['duration', 'event', 'residual', 'sex', 'age1']], duration_col='duration', event_col='event', robust=True)

        plt.figure(figsize=(10, 6))

        wf_dec = WeibullFitter()
        wf_acc = WeibullFitter()
        wb_rmst_vals = {}
        if len(grp_dec) > 0:
            wf_dec.fit(grp_dec["duration"], grp_dec["event"], label=f"Decelerated (n={len(grp_dec)})")
            wf_dec.plot_survival_function(ci_show=True)
            wb_rmst_vals['Decelerated'] = restricted_mean_survival_time(wf_dec, t=t_max)
        if len(grp_acc) > 0:
            wf_acc.fit(grp_acc["duration"], grp_acc["event"], label=f"Accelerated (n={len(grp_acc)})")
            wf_acc.plot_survival_function(ci_show=True)
            wb_rmst_vals['Accelerated'] = restricted_mean_survival_time(wf_acc, t=t_max)

        plt.title(f'Weibull Survival Fitter - {model_name} {fold_or_pooled}')
        plt.xlabel('Years')
        plt.ylabel('Survival Probability')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, max_duration + 1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'weibull_plot_{fold_or_pooled}.png'))
        plt.close('all')

        km_rmst_high = km_rmst_vals.get('Accelerated', np.nan)
        km_rmst_low  = km_rmst_vals.get('Decelerated', np.nan)
        wb_rmst_high = wb_rmst_vals.get('Accelerated', np.nan)
        wb_rmst_low  = wb_rmst_vals.get('Decelerated', np.nan)
        rmst_high = wb_rmst_high
        rmst_low  = wb_rmst_low
        rmst_diff = rmst_low - rmst_high
        
        summary_row_w = aft.summary.loc[('lambda_', 'residual')]
        res_weibull = {
            'Fold': fold_or_pooled,
            'Model': model_name,
            'Covariates': 'age1, sex, residual',
            'Strata': 'None',
            'Coef': summary_row_w['coef'],
            'TimeRatio': np.exp(summary_row_w['coef']),
            'TR_Lower_95': summary_row_w['exp(coef) lower 95%'],
            'TR_Upper_95': summary_row_w['exp(coef) upper 95%'],
            'P-value': summary_row_w['p'],
            'AIC': aft.AIC_,
            'Concordance': aft.concordance_index_,
            'RMST_Low': rmst_low,
            'RMST_High': rmst_high,
            'RMST_Diff_L_minus_H': rmst_diff,
            'N': len(df_results)
        }
        
        pd.DataFrame([res_full]).to_csv(os.path.join(csvs_dir, 'cox_full_summary.csv'), index=False)
        pd.DataFrame([res_strat_sex]).to_csv(os.path.join(csvs_dir, 'cox_strat_sex_summary.csv'), index=False)
        pd.DataFrame([res_raw_strat]).to_csv(os.path.join(csvs_dir, 'cox_raw_strat_summary.csv'), index=False)
        pd.DataFrame([res_std_strat]).to_csv(os.path.join(csvs_dir, 'cox_std_strat_summary.csv'), index=False)
        pd.DataFrame([res_weibull]).to_csv(os.path.join(csvs_dir, 'weibull_summary.csv'), index=False)
        
        return res_full, res_strat_sex, res_raw_strat, res_std_strat, res_weibull
        
    except Exception as e:
        print(f"    Cox Analysis Failed for {fold_or_pooled}: {e}")
        return None, None, None, None, None


def update_cox_summaries(exp_dir, disease_name):
    print(f"\nProcessing Experiment: {exp_dir} ({disease_name})")
    
    models = ['Single', 'Single_Delta', 'SiameseSubtract']
    summary_results_full = []
    summary_results_strat_sex = []
    summary_results_raw_strat = []
    summary_results_std_strat = []
    summary_results_weibull = []
    
    meta_df = None
    meta_file = f'tfrecords/{disease_name}_triplet/metadata.csv'
    if os.path.exists(meta_file):
        meta_df = pd.read_csv(meta_file, usecols=['ID', 'age1', 'sex'])
        meta_df['ID'] = meta_df['ID'].astype(str).str.zfill(8)

    model_fold_dfs = {m: {} for m in models}
    for model_name in models:
        for fold in range(5):
            pred_path = os.path.join(exp_dir, f'fold_{fold}', model_name, 'predictions.csv')
            if not os.path.exists(pred_path):
                print(f"    Warning: Missing predictions for fold {fold} model {model_name}")
                continue
                
            df_results = pd.read_csv(pred_path)
            if 'subject_id' in df_results.columns:
                df_results['subject_id'] = df_results['subject_id'].astype(str)
            
            if 'age1' not in df_results.columns or 'sex' not in df_results.columns:
                if meta_df is not None and 'subject_id' in df_results.columns:
                    df_results = pd.merge(df_results, meta_df, left_on='subject_id', right_on='ID', how='inner')
                    
            if 'age1' not in df_results.columns or 'sex' not in df_results.columns:
                 print(f"    Missing age1/sex for fold {fold}, model {model_name}. Skipping.")
                 continue
                 
            model_fold_dfs[model_name][fold] = df_results
            
    print("\n  [Pooled Analysis]")
    for model_name in models:
        fold_dfs = [model_fold_dfs[model_name][f] for f in range(5) if f in model_fold_dfs[model_name]]
        if len(fold_dfs) > 0:
            print(f"  Model: {model_name} (Pooled)")
            df_pooled = pd.concat(fold_dfs, ignore_index=True)
            res_full, res_strat_sex, res_raw_strat, res_std_strat, res_weibull = evaluate_survival(df_pooled, 'Pooled', model_name, exp_dir)
            if res_full: summary_results_full.append(res_full)
            if res_strat_sex: summary_results_strat_sex.append(res_strat_sex)
            if res_raw_strat: summary_results_raw_strat.append(res_raw_strat)
            if res_std_strat: summary_results_std_strat.append(res_std_strat)
            if res_weibull: summary_results_weibull.append(res_weibull)

    print("\n  [Fold-Specific Analysis]")
    for fold in range(5):
        print(f"  --- Fold {fold} ---")
        for model_name in models:
            if fold in model_fold_dfs[model_name]:
                print(f"  Model: {model_name}")
                df_results = model_fold_dfs[model_name][fold]
                res_full, res_strat_sex, res_raw_strat, res_std_strat, res_weibull = evaluate_survival(df_results, str(fold), model_name, exp_dir)
                if res_full: summary_results_full.append(res_full)
                if res_strat_sex: summary_results_strat_sex.append(res_strat_sex)
                if res_raw_strat: summary_results_raw_strat.append(res_raw_strat)
                if res_std_strat: summary_results_std_strat.append(res_std_strat)
                if res_weibull: summary_results_weibull.append(res_weibull)

    if summary_results_full:
        df_summary_full = pd.DataFrame(summary_results_full)
        df_summary_full.to_csv(os.path.join(exp_dir, 'cox_full_cv_summary.csv'), index=False)
        print(f"    Saved cox_full_cv_summary.csv")
        
    if summary_results_strat_sex:
        df_summary_strat_sex = pd.DataFrame(summary_results_strat_sex)
        df_summary_strat_sex.to_csv(os.path.join(exp_dir, 'cox_strat_sex_cv_summary.csv'), index=False)
        print(f"    Saved cox_strat_sex_cv_summary.csv")
        
    if summary_results_raw_strat:
        df_summary_raw_strat = pd.DataFrame(summary_results_raw_strat)
        df_summary_raw_strat.to_csv(os.path.join(exp_dir, 'cox_raw_strat_cv_summary.csv'), index=False)
        print(f"    Saved cox_raw_strat_cv_summary.csv")
    
    if summary_results_std_strat:
        df_summary_std_strat = pd.DataFrame(summary_results_std_strat)
        df_summary_std_strat.to_csv(os.path.join(exp_dir, 'cox_std_strat_cv_summary.csv'), index=False)
        print(f"    Saved cox_std_strat_cv_summary.csv")

    if summary_results_weibull:
        df_summary_weibull = pd.DataFrame(summary_results_weibull)
        df_summary_weibull.to_csv(os.path.join(exp_dir, 'weibull_cv_summary.csv'), index=False)
        print(f"    Saved weibull_cv_summary.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='all')
    args = parser.parse_args()

    all_diseases = ['hypertension', 'diabetes', 'dyslipidemia', 'obesity']
    if args.disease.lower() == 'all':
        diseases = all_diseases
    else:
        diseases = [args.disease]

    found_any = False
    
    for d in diseases:
        candidates = sorted(glob.glob(f"{d}_*"))
        valid_cands = [c for c in candidates if os.path.isdir(c) and len(c.rsplit('_', 1)) == 2 and c.rsplit('_', 1)[1].isdigit()]
        if valid_cands:
            latest_dir = valid_cands[-1]
            update_cox_summaries(latest_dir, d)
            found_any = True
                
    if not found_any:
        print("No experiment directories found matching expected pattern.")

if __name__ == "__main__":
    main()
