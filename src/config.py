# =========================
# DATASET
# =========================

DATASET_NAME = "physionet"
# "physionet"
# "bci_competition"


# =========================
# FEATURE EXTRACTION
# =========================

CSP_TYPE = "mne"
# "custom"
# "mne"

FEATURE_TYPE = "csp"
# "csp"
# "wavelets"
# "combined"


# =========================
# CLASSIFIER
# =========================

CLASSIFIER_TYPE = "custom_logistic"
# "lda"
# "logistic"
# "custom_logistic"


# =========================
# CSP PARAMETERS
# =========================

N_CSP_COMPONENTS = 4


# =========================
# WAVELET PARAMETERS
# =========================

WAVELET = "db4"
WAVELET_LEVEL = 4


# =========================
# LOGISTIC PARAMETERS
# =========================

LOGISTIC_LEARNING_RATE = 0.1
LOGISTIC_ITERATIONS = 2000

# =========================
# MAX SUBJECTS
# =========================

MAX_SUBJECTS = 15