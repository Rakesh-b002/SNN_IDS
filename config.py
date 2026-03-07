import os

BASE_DIR    = os.path.expanduser("~/snn_ids")
TON_IOT_DIR = os.path.join(BASE_DIR, "data/ton_iot")
CIC_IDS_DIR = os.path.join(BASE_DIR, "data/cic_ids")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SRC_DIR     = os.path.join(BASE_DIR, "src")

# SNN parameters (from paper)
N_INPUT   = 20
N_HIDDEN  = 40
N_OUTPUT  = 5
N_EPOCHS  = 30
T_WINDOW  = 100   # ms
DELTA     = 5     # ms refractory period
N_FEATURES = 20

# Classes
CLASSES = ['Normal', 'DoS', 'DDoS', 'Scan', 'Others']

print("Config loaded. Base dir:", BASE_DIR)
