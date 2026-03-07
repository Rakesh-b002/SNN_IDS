import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import joblib, os, glob, gc

CIC_DIR = os.path.expanduser("~/snn_ids/data/cic_ids/")
OUT_DIR = os.path.expanduser("~/snn_ids/data/processed/")
os.makedirs(OUT_DIR, exist_ok=True)

# ── FIXED label mapping (DDoS checked BEFORE DoS) ─────
def map_cic_label(x):
    x = str(x).lower().strip()
    if x in ['benign', 'normal']:               return 'Normal'
    if 'ddos' in x or 'bot' in x:              return 'DDoS'   # MUST be before DoS check
    if 'dos'  in x or 'hulk' in x or 'slowloris' in x: return 'DoS'
    if 'scan' in x or 'port' in x:             return 'Scan'
    return 'Others'

print("=== Processing CIC-IDS2018 (Fixed Label Mapping) ===")
cic_files = sorted(glob.glob(f"{CIC_DIR}*.csv"))
print(f"Found {len(cic_files)} files\n")

# ── Step 1: Find common numeric columns ───────────────
print("Finding common numeric columns across all files...")
col_sets = []
for f in cic_files:
    try:
        peek = pd.read_csv(f, nrows=5, encoding='utf-8', low_memory=False)
        peek.columns = peek.columns.str.strip()
        num_cols = set(peek.select_dtypes(include=[np.number]).columns)
        col_sets.append(num_cols)
        print(f"  {os.path.basename(f)}: {len(num_cols)} numeric cols")
    except Exception as e:
        print(f"  Could not peek {f}: {e}")

common_numeric = set.intersection(*col_sets)
print(f"\nCommon numeric columns: {len(common_numeric)}\n")

# ── Step 2: Identify which file contains PortScan ─────
# Thursday-22-02-2018 contains PortScan — load it separately
# with higher sample rate to guarantee Scan class is captured
SCAN_FILE = "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"

# ── Step 3: Load all files chunk by chunk ─────────────
all_samples  = []
CHUNK_SIZE   = 10000
KEEP_DEFAULT = 500   # rows kept per chunk for normal files
KEEP_SCAN    = 2000  # higher keep rate for Scan file

for f in cic_files:
    fname = os.path.basename(f)
    is_scan_file = SCAN_FILE in fname
    keep_n = KEEP_SCAN if is_scan_file else KEEP_DEFAULT

    try:
        print(f"Loading: {fname}  (keep={keep_n}/chunk)")
        file_samples = []
        chunks_read  = 0

        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE,
                                  encoding='utf-8', low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            label_candidates = [c for c in chunk.columns
                                 if 'label' in c.lower()]
            if not label_candidates:
                continue
            label_col = label_candidates[0]

            # Map labels FIRST (before dropping label column)
            chunk['class'] = chunk[label_col].apply(map_cic_label)

            # Keep only common numeric columns + class
            keep_cols = [c for c in common_numeric
                         if c in chunk.columns] + ['class']
            chunk = chunk[keep_cols]

            # Clean
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if len(chunk) == 0:
                continue

            # For scan file: stratified sample to preserve Scan rows
            if is_scan_file:
                scan_rows  = chunk[chunk['class'] == 'Scan']
                other_rows = chunk[chunk['class'] != 'Scan']
                sample_parts = []
                if len(scan_rows) > 0:
                    sample_parts.append(scan_rows)   # keep ALL scan rows
                if len(other_rows) > 0:
                    n = min(len(other_rows), keep_n)
                    sample_parts.append(other_rows.sample(n, random_state=42))
                if sample_parts:
                    sample = pd.concat(sample_parts, ignore_index=True)
                else:
                    continue
            else:
                sample = chunk.sample(
                    min(len(chunk), keep_n), random_state=42)

            file_samples.append(sample)
            chunks_read += 1
            if chunks_read >= 20:
                break
            del chunk
            gc.collect()

        if file_samples:
            file_df = pd.concat(file_samples, ignore_index=True)
            all_samples.append(file_df)
            class_dist = file_df['class'].value_counts().to_dict()
            print(f"  Kept: {file_df.shape}  |  {class_dist}")
            del file_samples, file_df
            gc.collect()

    except Exception as e:
        print(f"  Skipped {f}: {e}")

# ── Step 4: Combine ───────────────────────────────────
print("\nCombining all samples...")
df_cic = pd.concat(all_samples, ignore_index=True)
del all_samples
gc.collect()

df_cic.columns = df_cic.columns.str.strip()
print(f"Combined shape: {df_cic.shape}")
print("Class distribution:\n", df_cic['class'].value_counts())

# Verify all 5 classes present
expected = {'Normal', 'DoS', 'DDoS', 'Scan', 'Others'}
found    = set(df_cic['class'].unique())
missing  = expected - found
if missing:
    print(f"\nWARNING: Missing classes: {missing}")
    print("Scan file may not have had PortScan rows in sampled chunks.")
else:
    print("\nAll 5 classes confirmed present ✓")

# ── Step 5: Force numeric after concat ────────────────
X = df_cic.drop(columns=['class'], errors='ignore')
X = X.apply(pd.to_numeric, errors='coerce')
thresh = int(0.9 * len(X))
X = X.dropna(axis=1, thresh=thresh)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.clip(lower=-1e15, upper=1e15)

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    X = X.drop(columns=non_numeric)

print(f"\nShape after clean: {X.shape}")
print("All columns numeric ✓")

# ── Step 6: Encode labels ─────────────────────────────
y_raw = df_cic['class'].iloc[:len(X)].reset_index(drop=True)
le    = LabelEncoder()
y     = le.fit_transform(y_raw)
print(f"Final classes: {le.classes_}")

# ── Step 7: Feature selection ─────────────────────────
print(f"\nSelecting top 20 features from {X.shape[1]} columns...")
mi    = mutual_info_classif(X, y, random_state=42)
top20 = pd.Series(mi, index=X.columns).nlargest(20).index.tolist()
print(f"Top 20: {top20}")

# ── Step 8: Normalize + split ─────────────────────────
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X[top20])

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y)

# ── Step 9: Save ──────────────────────────────────────
np.save(f"{OUT_DIR}cic_X_train.npy", X_train)
np.save(f"{OUT_DIR}cic_X_test.npy",  X_test)
np.save(f"{OUT_DIR}cic_y_train.npy", y_train)
np.save(f"{OUT_DIR}cic_y_test.npy",  y_test)
joblib.dump(scaler, f"{OUT_DIR}cic_scaler.pkl")
joblib.dump(le,     f"{OUT_DIR}cic_label_encoder.pkl")
joblib.dump(top20,  f"{OUT_DIR}cic_top20_features.pkl")

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Val range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print("CIC-IDS2018 preprocessing complete ✓")

# ── Step 10: Verify saved files ───────────────────────
print("\n=== Saved files ===")
for fname in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, fname)) / 1024
    print(f"  {fname:45s} {size:8.1f} KB")
