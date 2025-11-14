"""
Path configuration for the project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_VIDEOS_DIR = DATA_DIR / 'raw_videos'
ENCODED_VIDEOS_DIR = DATA_DIR / 'encoded_videos'
VMAF_SCORES_DIR = DATA_DIR / 'vmaf_scores'
CONTENT_FEATURES_DIR = DATA_DIR / 'content_features'
NETWORK_TRACES_DIR = DATA_DIR / 'network_traces'
PROCESSED_TRACES_DIR = NETWORK_TRACES_DIR / 'processed'

# Results directories
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
LOGS_DIR = RESULTS_DIR / 'logs'

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_paths():
    """Get all paths as a dictionary."""
    return {
        'project_root': PROJECT_ROOT,
        'data_dir': DATA_DIR,
        'raw_videos': RAW_VIDEOS_DIR,
        'encoded_videos': ENCODED_VIDEOS_DIR,
        'vmaf_scores': VMAF_SCORES_DIR,
        'content_features': CONTENT_FEATURES_DIR,
        'network_traces': NETWORK_TRACES_DIR,
        'processed_traces': PROCESSED_TRACES_DIR,
        'results': RESULTS_DIR,
        'models': MODELS_DIR,
        'logs': LOGS_DIR
    }


if __name__ == '__main__':
    """Test paths."""
    print("\n📁 Project Paths:\n")
    paths = get_paths()
    for name, path in paths.items():
        exists = "✓" if path.exists() else "✗"
        print(f"{exists} {name:20s}: {path}")