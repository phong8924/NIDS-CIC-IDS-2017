import os

# --- Google Drive paths (Colab) ---
DRIVE_BASE = "/content/drive/MyDrive/DoAn_NIDS/Dataset/"
DRIVE_BINARY = os.path.join(DRIVE_BASE, "Binary_Data/")
DRIVE_MULTI = os.path.join(DRIVE_BASE, "Multi_Data/")
DRIVE_MODELS_DNN = os.path.join(DRIVE_BASE, "Models-DNN/")
DRIVE_MODELS_CNN = os.path.join(DRIVE_BASE, "Models-CNN/")

# --- Local paths (fallback) ---
LOCAL_DATA = "data/"
LOCAL_RESULTS = "results/"

# --- Data constants ---
N_FEATURES = 37

# --- Model hyperparameters (defaults) ---
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 0.001

# --- Callbacks ---
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-5

# --- Multiclass label mapping: 15 original -> 5 groups ---
MULTICLASS_MAPPING = {
    0: 0,   # Benign
    2: 1,   # DDoS
    3: 1,   # DoS GoldenEye
    4: 1,   # DoS Hulk
    5: 1,   # DoS Slowhttptest
    6: 1,   # DoS slowloris
    10: 2,  # PortScan
    7: 3,   # FTP-Patator
    11: 3,  # SSH-Patator
    12: 3,  # Web Attack - Brute Force
    1: 4,   # Bot
    8: 4,   # Heartbleed
    9: 4,   # Infiltration
    13: 4,  # Web Attack - Sql Injection
    14: 4,  # Web Attack - XSS
}

CLASS_NAMES_15 = [
    "Benign", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", "Heartbleed",
    "Infiltration", "PortScan", "SSH-Patator",
    "Web Attack - Brute Force", "Web Attack - Sql Injection", "Web Attack - XSS",
]

CLASS_NAMES_5 = ["Benign", "DoS/DDoS", "PortScan", "BruteForce", "Other"]
