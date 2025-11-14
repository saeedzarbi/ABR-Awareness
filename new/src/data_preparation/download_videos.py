"""
Download diverse video dataset for comprehensive evaluation.
"""

import urllib.request
from pathlib import Path
from typing import Dict, List
import subprocess


class DiverseVideoDownloader:
    """Download videos with different content characteristics."""
    
    # Curated diverse videos
    VIDEOS = {
        'bigbuckbunny': {
            'url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            'description': 'Animation - High detail, moderate motion',
            'expected_si': 'High (60-80)',
            'expected_ti': 'Medium (10-20)',
            'duration': '10 min',
            'size_mb': 158
        },
        'elephantsdream': {
            'url': 'http://ftp.nluug.nl/pub/graphics/blender/demo/movies/ED_1280.avi',
            'description': 'Surreal animation - Complex textures',
            'expected_si': 'High (70-90)',
            'expected_ti': 'Medium (15-25)',
            'duration': '11 min',
            'size_mb': 208
        },
        'tearsofsteel_short': {
            'url': 'https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov',
            'description': 'Live action - High motion scenes',
            'expected_si': 'Medium (40-60)',
            'expected_ti': 'High (20-40)',
            'duration': '12 min',
            'size_mb': 365
        },
        'sintel': {
            'url': 'https://download.blender.org/demo/movies/Sintel.2010.720p.mkv',
            'description': 'Animation - Fast camera movements',
            'expected_si': 'High (60-80)',
            'expected_ti': 'High (25-35)',
            'duration': '15 min',
            'size_mb': 432
        }
    }
    
    # Lightweight alternatives (if above fail)
    FALLBACK_VIDEOS = {
        'bbb_short': {
            'url': 'https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip',
            'description': 'Big Buck Bunny excerpt',
            'duration': '1 min',
            'size_mb': 50
        }
    }
    
    def __init__(self, output_dir: str = 'data/raw_videos'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_video(self, name: str, url: str, expected_size_mb: float) -> bool:
        """Download a single video."""
        # Support both direct files and archives
        if url.endswith('.zip'):
            output_path = self.output_dir / f"{name}.zip"
        elif url.endswith('.mov'):
            output_path = self.output_dir / f"{name}.mov"
        elif url.endswith('.mkv'):
            output_path = self.output_dir / f"{name}.mkv"
        elif url.endswith('.avi'):
            output_path = self.output_dir / f"{name}.avi"
        else:
            output_path = self.output_dir / f"{name}.mp4"
        
        if output_path.exists():
            print(f"✓ {name} already exists")
            return True
        
        print(f"⬇ Downloading {name} ({expected_size_mb} MB)...")
        
        try:
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded / total_size * 100, 100)
                print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, reporthook=progress)
            print(f"\n✓ Downloaded {name}")
            
            # Convert to mp4 if needed
            if not output_path.suffix == '.mp4':
                mp4_path = output_path.with_suffix('.mp4')
                print(f"  Converting to MP4...")
                subprocess.run([
                    'ffmpeg', '-i', str(output_path),
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-crf', '23', '-c:a', 'aac',
                    str(mp4_path), '-y'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                output_path.unlink()
                print(f"  ✓ Converted to {mp4_path.name}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_all(self, max_videos: int = 4) -> List[str]:
        """Download diverse video set."""
        print(f"\n{'='*70}")
        print(f"Downloading Diverse Video Dataset")
        print(f"Target: {max_videos} videos with content diversity")
        print(f"{'='*70}\n")
        
        successful = []
        
        for name, info in list(self.VIDEOS.items())[:max_videos]:
            print(f"\n📹 {name}")
            print(f"   {info['description']}")
            print(f"   Expected SI: {info['expected_si']}, TI: {info['expected_ti']}")
            
            if self.download_video(name, info['url'], info['size_mb']):
                successful.append(name)
            
            if len(successful) >= max_videos:
                break
        
        print(f"\n{'='*70}")
        print(f"Downloaded: {len(successful)}/{max_videos} videos")
        print(f"{'='*70}\n")
        
        return successful
    
    def verify_diversity(self) -> Dict:
        """Check SI/TI diversity of downloaded videos."""
        siti_dir = Path('data/content_features')
        
        if not siti_dir.exists():
            print("⚠ Run SI/TI extraction first")
            return {}
        
        videos = []
        for siti_file in siti_dir.glob('*_siti.json'):
            import json
            with open(siti_file) as f:
                data = json.load(f)
                videos.append({
                    'name': siti_file.stem.replace('_siti', ''),
                    'si': data['mean_si'],
                    'ti': data['mean_ti']
                })
        
        if videos:
            print("\n" + "="*70)
            print("Content Diversity Check:")
            print("="*70)
            print(f"{'Video':<20} | {'SI':>8} | {'TI':>8} | {'Category':<25}")
            print("-"*70)
            
            for v in videos:
                si, ti = v['si'], v['ti']
                
                if si > 60 and ti > 20:
                    cat = "High complexity + motion"
                elif si > 60:
                    cat = "High detail, low motion"
                elif ti > 20:
                    cat = "Low detail, high motion"
                else:
                    cat = "Simple content"
                
                print(f"{v['name']:<20} | {si:8.1f} | {ti:8.1f} | {cat:<25}")
            
            print("="*70 + "\n")
        
        return {'videos': videos}


def main():
    """Main download script."""
    print("\n🎬 Diverse Video Dataset Downloader\n")
    
    downloader = DiverseVideoDownloader()
    
    # Download
    num = input("How many videos to download? (recommended: 4): ").strip()
    num_videos = int(num) if num else 4
    
    successful = downloader.download_all(max_videos=num_videos)
    
    if successful:
        print("\n✓ Videos downloaded!")
        print("\nNext steps:")
        print("  1. Encode videos: python src/data_preparation/video_encoder.py")
        print("  2. Calculate VMAF: python src/data_preparation/vmaf_calculator.py")
        print("  3. Extract SI/TI: python src/data_preparation/si_ti_extractor.py")
    else:
        print("\n✗ No videos downloaded")


if __name__ == '__main__':
    main()