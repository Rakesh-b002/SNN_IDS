import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import joblib, os, glob

# ── Paths ──────────────────────────────────────────────
TON_PATH  = os.path.expanduser("~/snn_ids/data/ton_iot/train_test_network.csv")
CIC_DIR   = os.path.expanduser("~/snn_ids/data/cic_ids/")
OUT_DIR   = os.path.expanduser("~/snn_ids/data/processed/")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Label mapping (paper's 5 classes) ──────────────────
label_map = {
    'normal'    : 'Normal',
    'dos'       : 'DoS',
    'ddos'      : 'DDoS',
    'scanning'  : 'Scan',
    'backdoor'  : 'Others',
    'injection' : 'Others',
    'password'  : 'Others',
    'ransomware': 'Others',
    'xss'       : 'Others',
    'mitm'      : 'Others',
}

def clean(df, label_col):
    print(f"  Before cleaning : {df.shape}")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"  After cleaning  : {df.shape}")
    return df

def select_features(X, y, n=20):
    print(f"  Selecting top {n} features via Mutual Information...")
    mi = mutual_info_classif(X, y, random_state=42)
    top = pd.Series(mi, index=X.columns).nlargest(n).index.tolist()
    print(f"  Top {n} features: {top}")
    return top

# ── Process TON_IoT ────────────────────────────────────
print("\n=== Processing TON_IoT ===")
df = pd.read_csv(TON_PATH)
print(f"  Loaded: {df.shape}")

# Map labels
df['class'] = df['type'].str.lower().map(label_map)
print("  Class distribution:\n", df['class'].value_counts())

# Drop non-numeric and label columns
drop_cols = ['type', 'class', 'src_ip', 'dst_ip',
             'dns_query', 'ssl_subject', 'ssl_issuer',
             'http_uri', 'http_user_agent', 'weird_name',
             'weird_addl', 'weird_notice', 'label']
df_clean = clean(df, 'class')
X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])
y_raw = df_clean['class']

le = LabelEncoder()
y = le.fit_transform(y_raw)
print("  Classes:", le.classes_)

# Feature selection
top20 = select_features(X, y)
X_sel = X[top20]

# Normalize
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X_sel)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y)

# Save
np.save(f"{OUT_DIR}ton_X_train.npy", X_train)
np.save(f"{OUT_DIR}ton_X_test.npy",  X_test)
np.save(f"{OUT_DIR}ton_y_train.npy", y_train)
np.save(f"{OUT_DIR}ton_y_test.npy",  y_test)
joblib.dump(scaler, f"{OUT_DIR}ton_scaler.pkl")
joblib.dump(le,     f"{OUT_DIR}ton_label_encoder.pkl")
joblib.dump(top20,  f"{OUT_DIR}ton_top20_features.pkl")

print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
print("  TON_IoT preprocessing complete ✓")

# ── Process CIC-IDS2018 ────────────────────────────────
print("\n=== Processing CIC-IDS2018 ===")
cic_files = glob.glob(f"{CIC_DIR}*.csv")
print(f"  Found {len(cic_files)} CSV files")

dfs = []
for f in cic_files:
    try:
        tmp = pd.read_csv(f, encoding='utf-8', low_memory=False)
        dfs.append(tmp)
        print(f"  Loaded: {os.path.basename(f)} → {tmp.shape}")
    except Exception as e:
        print(f"  Skipped {f}: {e}")

df_cic = pd.concat(dfs, ignore_index=True)
print(f"  Combined shape: {df_cic.shape}")

# Find label column
label_col = [c for c in df_cic.columns if 'label' in c.lower()][0]
print(f"  Label column: {label_col}")
print("  Raw labels:\n", df_cic[label_col].value_counts().head(10))

# Map CIC labels to 5 classes
def map_cic_label(x):
    x = str(x).lower().strip()
    if x in ['benign', 'normal']: return 'Normal'
    if 'dos'  in x or 'hulk' in x or 'slowloris' in x: return 'DoS'
    if 'ddos' in x or 'bot'  in x: return 'DDoS'
    if 'scan' in x or 'port' in x: return 'Scan'
    return 'Others'

df_cic['class'] = df_cic[label_col].apply(map_cic_label)
print("  Mapped classes:\n", df_cic['class'].value_counts())

df_cic_clean = clean(df_cic, 'class')
X_cic = df_cic_clean.drop(columns=[label_col, 'class'], errors='ignore')
X_cic = X_cic.select_dtypes(include=[np.number])
y_cic_raw = df_cic_clean['class']

le_cic = LabelEncoder()
y_cic = le_cic.fit_transform(y_cic_raw)

top20_cic = select_features(X_cic, y_cic)
X_cic_sel = X_cic[top20_cic]

scaler_cic = MinMaxScaler()
X_cic_norm = scaler_cic.fit_transform(X_cic_sel)

X_cic_train, X_cic_test, y_cic_train, y_cic_test = train_test_split(
    X_cic_norm, y_cic, test_size=0.2, random_state=42, stratify=y_cic)

np.save(f"{OUT_DIR}cic_X_train.npy", X_cic_train)
np.save(f"{OUT_DIR}cic_X_test.npy",  X_cic_test)
np.save(f"{OUT_DIR}cic_y_train.npy", y_cic_train)
np.save(f"{OUT_DIR}cic_y_test.npy",  y_cic_test)
joblib.dump(scaler_cic, f"{OUT_DIR}cic_scaler.pkl")
joblib.dump(le_cic,     f"{OUT_DIR}cic_label_encoder.pkl")
joblib.dump(top20_cic,  f"{OUT_DIR}cic_top20_features.pkl")

print(f"  Train: {X_cic_train.shape} | Test: {X_cic_test.shape}")
print("  CIC-IDS2018 preprocessing complete ✓")

print("\n=== ALL DONE ===")
print(f"Processed files saved to: {OUT_DIR}")
print("Files saved:")
for f in os.listdir(OUT_DIR):
    print(f"  {f}")
