"""Configuration file for ABR Content-Aware project"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
VMAF_DIR = os.path.join(DATA_DIR, 'vmaf')
TRACES_DIR = os.path.join(DATA_DIR, 'traces')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Video settings
BITRATE_LEVELS = [300, 750, 1850, 2850, 4300, 6000]  # kbps
VIDEO_CHUNK_LEN = 4.0  # seconds
TOTAL_VIDEO_CHUNKS = 48  # per video
BUFFER_THRESH = 60.0  # seconds

# Content feature settings
CONTENT_FEATURE_DIM = 2  # SI, TI
VMAF_DIM = len(BITRATE_LEVELS)  # VMAF prediction for each bitrate

# Model settings
STATE_DIM = [6, 8]  # [features, time_steps] - same as Pensieve
ACTION_DIM = len(BITRATE_LEVELS)
LEARNING_RATE = 1e-4
GAMMA = 0.99  # discount factor

# Training settings
NUM_EPOCHS = 10000
NUM_WORKERS = 4
SAVE_INTERVAL = 100  # save model every N epochs
LOG_INTERVAL = 10

# Reward weights
QUALITY_WEIGHT = 1.0
REBUFFER_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
CONTENT_BONUS_WEIGHT = 0.5  # NEW

# GCS settings (optional)
GCS_BUCKET = None  # Set to your bucket name if using GCS
GCS_PREFIX = 'abr-research'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
