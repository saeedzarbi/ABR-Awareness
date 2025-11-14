"""
Download standard video datasets for ABR research.
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, List
import hashlib


class VideoDownloader:
    """Download and verify standard test videos."""
    
    VIDEOS: Dict[str, Dict[str, str]] = {
        'bigbuckbunny': {
            'url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            'duration': '9:56',
            'description': 'Animation with high motion and colorful scenes',
            'size_mb': 158
        },
        'sintel': {
            'url': 'http://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.1080p.mkv',
            'duration': '14:48',
            'description': 'Animation with fast camera movements',
            'size_mb': 675
        },
        'tearsofsteel': {
            'url': 'http://ftp.nluug.nl/pub/graphics/blender/demo/movies/ToS/ToS-4k-1920.mov',
            'duration': '12:14',
            'description': 'Live action with dark and bright scenes',
            'size_mb': 738
        },
        'elephantsdream': {
            'url': 'http://ftp.nluug.nl/pub/graphics/blender/demo/movies/ED_1280.avi',
            'duration': '10:53',
            'description': 'Surreal animation with complex textures',
            'size_mb': 208
        }
    }
    
    # Backup: lighter videos if download fails
    FALLBACK_VIDEOS: Dict[str, Dict[str, str]] = {
        'sample1': {
            'url': 'https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4',
            'duration': '0:10',
            'description': 'Short test clip',
            'size_mb': 1
        }
    }
    
    def __init__(self, output_dir: str = 'data/raw_videos'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_video(self, name: str, url: str, expected_size_mb: float) -> bool:
        """
        Download a single video with progress tracking.
        
        Args:
            name: Video identifier
            url: Download URL
            expected_size_mb: Expected file size in MB
            
        Returns:
            True if download successful
        """
        output_path = self.output_dir / f"{name}.mp4"
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            if abs(file_size_mb - expected_size_mb) < 5:  # 5MB tolerance
                print(f"✓ {name} already exists ({file_size_mb:.1f} MB)")
                return True
            else:
                print(f"⚠ {name} exists but size mismatch, re-downloading...")
                output_path.unlink()
        
        print(f"⬇ Downloading {name} ({expected_size_mb} MB)...")
        
        try:
            def progress_hook(block_num: int, block_size: int, total_size: int):
                downloaded = block_num * block_size
                percent = min(downloaded / total_size * 100, 100)
                print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
            print(f"\n✓ Downloaded {name} successfully")
            return True
            
        except Exception as e:
            print(f"\n✗ Failed to download {name}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_all(self, use_fallback: bool = False) -> List[str]:
        """
        Download all videos.
        
        Args:
            use_fallback: Use lightweight fallback videos if True
            
        Returns:
            List of successfully downloaded video names
        """
        videos_to_download = self.FALLBACK_VIDEOS if use_fallback else self.VIDEOS
        
        print(f"\n{'='*60}")
        print(f"Starting download of {len(videos_to_download)} videos")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}\n")
        
        successful = []
        failed = []
        
        for name, info in videos_to_download.items():
            print(f"\n[{len(successful)+len(failed)+1}/{len(videos_to_download)}] {name}")
            print(f"  Description: {info['description']}")
            print(f"  Duration: {info['duration']}")
            
            if self.download_video(name, info['url'], info['size_mb']):
                successful.append(name)
            else:
                failed.append(name)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  ✓ Successful: {len(successful)}/{len(videos_to_download)}")
        if failed:
            print(f"  ✗ Failed: {failed}")
        print(f"{'='*60}\n")
        
        return successful
    
    def verify_videos(self) -> Dict[str, bool]:
        """Verify downloaded videos exist and are readable."""
        results = {}
        
        for video_file in self.output_dir.glob("*.mp4"):
            try:
                size_mb = video_file.stat().st_size / (1024 * 1024)
                results[video_file.stem] = size_mb > 0.1  # At least 100KB
            except Exception:
                results[video_file.stem] = False
        
        return results


def main():
    """Main download script."""
    downloader = VideoDownloader(output_dir='data/raw_videos')
    
    print("\n🎬 Video Download Manager for ABR Research")
    print("\nOptions:")
    print("  1. Download full-size videos (~2GB total, recommended)")
    print("  2. Download lightweight test videos (~10MB total, for testing)")
    
    choice = input("\nYour choice (1 or 2): ").strip()
    
    use_fallback = choice == '2'
    
    successful = downloader.download_all(use_fallback=use_fallback)
    
    if successful:
        print("\n✓ Ready to proceed to encoding!")
        print(f"  Videos saved in: {downloader.output_dir.absolute()}")
        print("\nNext step: Run video encoding")
        print("  python src/data_preparation/video_encoder.py")
    else:
        print("\n✗ No videos downloaded successfully")
        print("  Please check your internet connection and try again")


if __name__ == '__main__':
    main()