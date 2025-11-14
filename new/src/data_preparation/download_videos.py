"""
Download one more video to complete the set.
"""

import urllib.request
from pathlib import Path
import subprocess


def download_video(name: str, url: str, output_dir: str = 'data/raw_videos'):
    """Download a single video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{name}.mp4"
    
    if output_path.exists():
        print(f"✓ {name} already exists")
        return True
    
    print(f"⬇ Downloading {name}...")
    
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=progress)
        print(f"\n✓ Downloaded {name}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    """Download a supplementary video."""
    
    # Alternative videos
    videos = {
        'crowd_run': {
            'url': 'https://media.xiph.org/video/derf/y4m/crowd_run_1080p50.y4m',
            'description': 'High motion crowd scene (WARNING: Large file ~500MB)',
        },
        'parkjoy': {
            'url': 'https://media.xiph.org/video/derf/y4m/park_joy_1080p50.y4m',
            'description': 'Outdoor scene with moderate motion (~200MB)',
        },
        'ducks': {
            'url': 'https://download.blender.org/demo/test/ducks_take_off_1080p.mp4',
            'description': 'Nature scene with animal motion',
        }
    }
    
    print("\n🎬 Download Additional Video\n")
    print("Available videos:")
    for i, (name, info) in enumerate(videos.items(), 1):
        print(f"  {i}. {name}: {info['description']}")
    
    choice = input("\nSelect video (1-3) or press Enter for default: ").strip()
    
    if choice == '2':
        selected = 'parkjoy'
    elif choice == '3':
        selected = 'ducks'
    else:
        selected = 'crowd_run'
    
    video_info = videos[selected]
    
    print(f"\nDownloading: {selected}")
    success = download_video(selected, video_info['url'])
    
    if success:
        print(f"\n✓ Done! Now you have 4 videos")
        print("\nNext step: Process all videos")
        print("  python src/data_preparation/video_encoder.py")


if __name__ == '__main__':
    main()