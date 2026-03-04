import os
import glob
import json
import argparse
import sys
import random
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_dir = 'pred_model'
config_path = os.path.join(model_dir, 'config.json')
model_path = os.path.join(model_dir, 'model.pth')

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    print(f"Error: Config file not found in {config_path}")
    sys.exit(1)

sys.path.append(os.path.abspath('pred_model'))
try:
    from resnet import ResNet1d
except ImportError:
    print("Error: Could not import ResNet1d from pred_model.")
    sys.exit(1)

def process_tfrecord_signal(signal_8lead):
    sig = signal_8lead.T
    l1, l2, v1_v6 = sig[0], sig[1], sig[2:]
    ecg_12lead = np.vstack([l1, l2, l2 - l1, -0.5 * (l1 + l2), l1 - 0.5 * l2, l2 - 0.5 * l1, v1_v6])
    resampled = resample(ecg_12lead, 4000, axis=1)
    return np.pad(resampled, ((0, 0), (48, 48)), mode='constant')

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class SiameseAgeResNetSubtract(nn.Module):
    def __init__(self, backbone):
        super(SiameseAgeResNetSubtract, self).__init__()
        self.backbone = backbone
        self.feature_dim = backbone.lin.in_features
        self.backbone.lin = nn.Identity()
        self.fusion = nn.Linear(self.feature_dim, 1)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)
        diff = feat2 - feat1
        return self.fusion(diff)

def get_baseline_model():
    model = ResNet1d(input_dim=(12, config['seq_length']),
                     blocks_dim=list(zip(config['net_filter_size'], config['net_seq_length'])),
                     n_classes=1, kernel_size=config['kernel_size'], dropout_rate=config['dropout_rate'])
    cp = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model'])
    return model.to(device)

def get_siamese_subtract_model():
    b = ResNet1d(input_dim=(12, config['seq_length']),
                 blocks_dim=list(zip(config['net_filter_size'], config['net_seq_length'])),
                 n_classes=1, kernel_size=config['kernel_size'], dropout_rate=config['dropout_rate'])
    cp = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = {k: v for k, v in cp['model'].items() if not k.startswith('lin.')}
    b.load_state_dict(state_dict, strict=False)
    model = SiameseAgeResNetSubtract(b)
    return model.to(device)

class AgeDeltaDataset(Dataset):
    def __init__(self, file_patterns, DISEASE, is_train=False, task='siamese', dry_run=False):
        self.samples = []
        csv_path = f'tfrecords/{DISEASE}_triplet/metadata.csv'
        df_meta = pd.read_csv(csv_path, low_memory=False)
            
        df_meta['ID'] = df_meta['ID'].astype(str).str.zfill(8)

        for c in ['dt0', 'dt1', 'dt2']:
            if c in df_meta.columns:
                df_meta[c] = pd.to_datetime(df_meta[c], errors='coerce')
        
        df_meta['dt0_str'] = df_meta['dt0'].dt.strftime('%Y-%m-%d')
        df_meta['dt1_str'] = df_meta['dt1'].dt.strftime('%Y-%m-%d')
        df_meta['key'] = df_meta['ID'] + '_' + df_meta['dt0_str'] + '_' + df_meta['dt1_str']

        label_map = {}
        target_event_col = f'{DISEASE}2'
        
        for _, row in df_meta.iterrows():
            dur = np.nan
            if target_event_col in df_meta.columns and pd.notnull(row.get('dt1')) and pd.notnull(row.get('dt2')):
                dur = (row['dt2'] - row['dt1']).days / 365.25
                
            label_map[row['key']] = {
                'dt0': row.get(f'{DISEASE}_dt0', 0),
                'dt1': row.get(f'{DISEASE}_dt1', 0),
                'disease2': row.get(target_event_col, np.nan),
                'duration': dur,
                'age0': row.get('age0', np.nan),
                'age1': row.get('age1', np.nan),
                'sex': 1 if str(row.get('sex')).upper().startswith('M') else 0,
                'SUBJECT_ID': row['ID']
            }

        for file_path in file_patterns:
            raw_dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
            if dry_run: raw_dataset = raw_dataset.take(50)
            
            for raw_record in raw_dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                f = example.features.feature

                pid = f['pid'].bytes_list.value[0].decode()
                dt0 = f['dt0'].bytes_list.value[0].decode().split(' ')[0]
                dt1 = f['dt1'].bytes_list.value[0].decode().split(' ')[0]
                key = f"{pid}_{dt0}_{dt1}"

                # Read T0 and T1
                age0 = f['age0'].float_list.value[0]
                age1 = f['age1'].float_list.value[0]
                sig0 = tf.io.parse_tensor(f['b_signal0'].bytes_list.value[0], out_type=tf.float32).numpy()
                sig1 = tf.io.parse_tensor(f['b_signal1'].bytes_list.value[0], out_type=tf.float32).numpy()

                # Try reading T2
                has_t2 = False
                sig2 = None
                age2 = None
                try:
                    if 'b_signal2' in f and 'age2' in f:
                         sig2 = tf.io.parse_tensor(f['b_signal2'].bytes_list.value[0], out_type=tf.float32).numpy()
                         age2 = f['age2'].float_list.value[0]
                         has_t2 = True
                except:
                    pass

                meta = label_map.get(key, {})
                if not meta:
                    continue
                
                meta['age0'] = age0
                meta['age1'] = age1
                meta['age2'] = age2

                if task == 'baseline':
                    if is_train:
                        if not np.isnan(age0) and not np.isnan(sig0).any():
                            self.samples.append({'sig': process_tfrecord_signal(sig0), 'age': age0})
                        if not np.isnan(age1) and not np.isnan(sig1).any():
                            self.samples.append({'sig': process_tfrecord_signal(sig1), 'age': age1})
                        if has_t2 and not np.isnan(age2) and not np.isnan(sig2).any():
                            self.samples.append({'sig': process_tfrecord_signal(sig2), 'age': age2})
                    else:
                        self.samples.append({
                            'sig0': process_tfrecord_signal(sig0) if not np.isnan(sig0).any() else None,
                            'sig1': process_tfrecord_signal(sig1) if not np.isnan(sig1).any() else None,
                            'age0': age0,
                            'age1': age1,
                            'metadata': meta
                        })
                elif task == 'siamese':
                    if not np.isnan(age0) and not np.isnan(age1) and not np.isnan(sig0).any() and not np.isnan(sig1).any():
                        self.samples.append({
                            'sig0': process_tfrecord_signal(sig0),
                            'sig1': process_tfrecord_signal(sig1),
                            'target': age1 - age0,
                            'metadata': meta
                        })

        self.is_train = is_train
        self.task = task

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.task == 'baseline':
            if self.is_train:
                return torch.tensor(s['sig'], dtype=torch.float32), torch.tensor([s['age']], dtype=torch.float32)
            else:
                return s
        else:
            sig0, sig1, target, meta = s['sig0'], s['sig1'], s['target'], s['metadata']
            if self.is_train and np.random.rand() > 0.5:
                return torch.tensor(sig1, dtype=torch.float32), torch.tensor(sig0, dtype=torch.float32), torch.tensor([-target], dtype=torch.float32), meta
            return torch.tensor(sig0, dtype=torch.float32), torch.tensor(sig1, dtype=torch.float32), torch.tensor([target], dtype=torch.float32), meta

def baseline_eval_collate(batch):
    return batch

def run_fold(disease, fold, train_files, val_files, test_files, exp_dir, epochs, early_stop_patience, dry_run=False):
    print(f"\n--- Starting Fold {fold} ---")
    
    fold_dir = os.path.join(exp_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    bl_train_ds = AgeDeltaDataset(train_files, disease, is_train=True, task='baseline', dry_run=dry_run)
    bl_val_ds = AgeDeltaDataset(val_files, disease, task='baseline', dry_run=dry_run)
    
    sia_train_ds = AgeDeltaDataset(train_files, disease, is_train=True, task='siamese', dry_run=dry_run)
    sia_val_ds = AgeDeltaDataset(val_files, disease, task='siamese', dry_run=dry_run)
    test_ds = AgeDeltaDataset(test_files, disease, task='siamese', dry_run=dry_run)
    
    valid_test_samples = [s for s in test_ds.samples if pd.notnull(s['metadata']['duration']) and pd.notnull(s['metadata']['disease2'])]
    if len(valid_test_samples) == 0:
        print(f"[Warning] No valid survival labels (disease2, duration) found in test set for Fold {fold}. Survival analysis will fail.")
    else:
        test_ds.samples = valid_test_samples

    bl_train_loader = DataLoader(bl_train_ds, batch_size=32, shuffle=True)
    bl_val_loader = DataLoader(bl_val_ds, batch_size=32, collate_fn=baseline_eval_collate)
    
    sia_train_loader = DataLoader(sia_train_ds, batch_size=32, shuffle=True)
    sia_val_loader = DataLoader(sia_val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=baseline_eval_collate)
    
    models_to_run = ['Single', 'SiameseSubtract']
    criterion = nn.MSELoss()
    
    for model_name in models_to_run:
        print(f"\n[Fold {fold}] Processing Model: {model_name}")
        model_out_dir = os.path.join(fold_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        model_save_path = os.path.join(model_out_dir, 'best_model.pt')
        
        if model_name == 'Single':
            model = get_baseline_model()
            for param in model.parameters(): param.requires_grad = False
            for param in model.lin.parameters(): param.requires_grad = True
            optimizer = optim.Adam(model.lin.parameters(), lr=0.001)
            train_loader = bl_train_loader
            val_loader = bl_val_loader
        else:
            model = get_siamese_subtract_model()
            for param in model.backbone.parameters(): param.requires_grad = False
            for param in model.fusion.parameters(): param.requires_grad = True
            optimizer = optim.Adam(model.fusion.parameters(), lr=0.001)
            train_loader = sia_train_loader
            val_loader = sia_val_loader

        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=model_save_path)
        
        for epoch in range(epochs):
            model.train()
            t_loss = 0
            for batch in train_loader:
                if model_name == 'Single':
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred = model(x)
                else:
                    x1, x2, y, meta = batch
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred = model(x1, x2)
                    
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            model.eval()
            v_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if model_name == 'Single':
                        sig1s_valid = []
                        ages1_valid = []
                        for item in batch:
                            if item['sig1'] is not None and not np.isnan(item['age1']):
                                sig1s_valid.append(torch.tensor(item['sig1'], dtype=torch.float32))
                                ages1_valid.append(item['age1'])
                                
                        if sig1s_valid:
                            x = torch.stack(sig1s_valid).to(device)
                            y = torch.tensor(ages1_valid, dtype=torch.float32).unsqueeze(1).to(device)
                            pred = model(x)
                            v_loss += criterion(pred, y).item()
                    else:
                        x1, x2, y, meta = batch
                        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                        pred = model(x1, x2)
                        v_loss += criterion(pred, y).item()
                    
            t_loss /= max(len(train_loader), 1)
            v_loss /= max(len(val_loader), 1)
            print(f"[{model_name}] Epoch {epoch+1}/{epochs}: Train MSE {t_loss:.4f}, Val MSE {v_loss:.4f}")
            
            early_stopping(v_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=False))
        
        model.eval()
        results = []
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 0: continue
                
                if model_name == 'Single':
                    sig0s_list = []
                    sig1s_list = []
                    valid_indices = []
                    for i, item in enumerate(batch):
                        if item[0] is not None and item[1] is not None:
                            sig0s_list.append(torch.tensor(item[0], dtype=torch.float32))
                            sig1s_list.append(torch.tensor(item[1], dtype=torch.float32))
                            valid_indices.append(i)
                            
                    if not valid_indices: continue
                    sig0s = torch.stack(sig0s_list).to(device)
                    sig1s = torch.stack(sig1s_list).to(device)

                    preds0 = model(sig0s).cpu().numpy().flatten()
                    preds1 = model(sig1s).cpu().numpy().flatten()
                    
                    for list_idx, batch_idx in enumerate(valid_indices):
                        item = batch[batch_idx]
                        meta = item[3]
                        age0, age1 = meta['age0'], meta['age1']
                        r0 = preds0[list_idx] - age0
                        r1 = preds1[list_idx] - age1
                        results.append({
                            'subject_id': meta['SUBJECT_ID'],
                            'age0': age0, 'age1': age1,
                            'sex': meta['sex'],
                            'residual0': r0,
                            'residual': r1,
                            'pred_delta': preds1[list_idx] - preds0[list_idx],
                            'actual_delta': age1 - age0,
                            'duration': meta['duration'],
                            'event': meta['disease2']
                        })
                else:
                    sig0s_list = []
                    sig1s_list = []
                    valid_indices = []
                    for i, item in enumerate(batch):
                        is_dict = isinstance(item, dict)
                        sig0_valid = item.get('sig0') is not None if is_dict else item[0] is not None
                        sig1_valid = item.get('sig1') is not None if is_dict else item[1] is not None
                        if sig0_valid and sig1_valid:
                            if is_dict:
                                sig0s_list.append(torch.tensor(item['sig0'], dtype=torch.float32))
                                sig1s_list.append(torch.tensor(item['sig1'], dtype=torch.float32))
                            else:
                                sig0s_list.append(torch.tensor(item[0], dtype=torch.float32))
                                sig1s_list.append(torch.tensor(item[1], dtype=torch.float32))
                            valid_indices.append(i)
                            
                    if not valid_indices: continue
                    sig0s = torch.stack(sig0s_list).to(device)
                    sig1s = torch.stack(sig1s_list).to(device)
                    preds = model(sig0s, sig1s).cpu().numpy().flatten()
                    
                    for list_idx, batch_idx in enumerate(valid_indices):
                        item = batch[batch_idx]
                        if isinstance(item, dict):
                            meta = item['metadata']
                            actual_delta = item['target'] if 'target' in item else (item['age1'] - item['age0'])
                        else:
                            meta = item[3]
                            actual_delta = item[2].item()
                            
                        pred_delta = preds[list_idx]
                        
                        results.append({
                            'subject_id': meta['SUBJECT_ID'],
                            'age0': meta['age0'], 'age1': meta['age1'],
                            'sex': meta['sex'],
                            'residual': pred_delta - actual_delta,
                            'pred_delta': pred_delta,
                            'actual_delta': actual_delta,
                            'duration': meta['duration'],
                            'event': meta['disease2']
                        })
                        
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(model_out_dir, 'predictions.csv'), index=False)
        print(f"Saved predictions to {model_out_dir}/predictions.csv")
    
    print(f"\n[Fold {fold}] Processing Model: Single_Delta")
    single_out_dir = os.path.join(fold_dir, 'Single')
    delta_out_dir = os.path.join(fold_dir, 'Single_Delta')
    os.makedirs(delta_out_dir, exist_ok=True)
    
    df_single = pd.read_csv(os.path.join(single_out_dir, 'predictions.csv'))
    df_delta = df_single.copy()
    
    df_delta['residual'] = df_delta['pred_delta'] - df_delta['actual_delta']
    df_delta.to_csv(os.path.join(delta_out_dir, 'predictions.csv'), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='all')
    parser.add_argument('--model_type', type=str, default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    diseases = ['hypertension', 'diabetes', 'dyslipidemia', 'obesity'] if args.disease.lower() == 'all' else [args.disease]

    for disease in diseases:
        print(f"Starting CV Pipeline for {disease}")
        
        data_dir = f'tfrecords/{disease}_triplet'
        if not os.path.exists(data_dir):
            print(f"Skipping {disease}: {data_dir} not found.")
            continue
            
        tfrecord_files = sorted(glob.glob(f'{data_dir}/*.tfrecords'))
        if len(tfrecord_files) < 10:
            print(f"Error: Found only {len(tfrecord_files)} shards in {data_dir}. Need 10 for 5-fold CV.")
            continue

        exp_name = f"{disease}_{datetime.now().strftime('%y%m%d%H%M%S')}"
        if args.dry_run: exp_name += "_dryrun"
        os.makedirs(exp_name, exist_ok=True)
        
        n_shards = min(10, len(tfrecord_files))
        
        for fold in range(5):
            test_shards = [(2*fold)%n_shards, (2*fold+1)%n_shards]
            val_shards = [(2*fold+2)%n_shards, (2*fold+3)%n_shards]
            train_shards = [s for s in range(n_shards) if s not in test_shards and s not in val_shards]
            
            train_files = [tfrecord_files[i] for i in train_shards]
            val_files = [tfrecord_files[i] for i in val_shards]
            test_files = [tfrecord_files[i] for i in test_shards]
            
            run_fold(disease, fold, train_files, val_files, test_files, exp_name, args.epochs, args.early_stop_patience, args.dry_run)
            if args.dry_run and fold == 1:
                break
                
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        print("Interrupted by KeyboardInterrupt")
    except Exception as e:
        import traceback
        traceback.print_exc()
