# =========================
# DATASET
# =========================

DATASET_NAME = "bci_competition"
# "physionet"
# "bci_competition"

# PhysioNet experiment runs
RUNS = (3, 4, 7, 8, 11, 12)


# =========================
# FEATURE EXTRACTION
# =========================

CSP_TYPE = "custom"
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
# MAX SUBJECTS
# =========================

MAX_SUBJECTS = 10