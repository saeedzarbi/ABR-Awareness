"""
Calculate VMAF scores for encoded videos.
VMAF: Video Multimethod Assessment Fusion - perceptual video quality metric
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil


class VMAFCalculator:
    """Calculate VMAF scores comparing encoded videos to reference."""
    
    def __init__(
        self,
        reference_dir: str = 'data/raw_videos',
        encoded_dir: str = 'data/encoded_videos',
        output_dir: str = 'data/vmaf_scores'
    ):
        self.reference_dir = Path(reference_dir)
        self.encoded_dir = Path(encoded_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check FFmpeg with libvmaf
        if not self._check_vmaf_support():
            raise RuntimeError(
                "FFmpeg with libvmaf not found!\n"
                "Install: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)"
            )
    
    def _check_vmaf_support(self) -> bool:
        """Check if FFmpeg supports VMAF filter."""
        if not shutil.which('ffmpeg'):
            return False
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-filters'],
                capture_output=True,
                text=True,
                check=True
            )
            return 'libvmaf' in result.stdout
        except:
            return False
    
    def calculate_vmaf(
        self,
        reference_video: Path,
        distorted_video: Path,
        output_json: Path
    ) -> Optional[float]:
        """
        Calculate VMAF score between reference and distorted video.
        
        Args:
            reference_video: Original high-quality video
            distorted_video: Encoded video to evaluate
            output_json: Path to save detailed VMAF output
            
        Returns:
            Mean VMAF score (0-100) or None if failed
        """
        # VMAF filter with model path
        vmaf_filter = (
            f"[0:v]setpts=PTS-STARTPTS[reference];"
            f"[1:v]setpts=PTS-STARTPTS[distorted];"
            f"[distorted][reference]libvmaf="
            f"log_fmt=json:"
            f"log_path={output_json}:"
            f"n_threads=4"
        )
        
        cmd = [
            'ffmpeg',
            '-i', str(distorted_video),    # Distorted first
            '-i', str(reference_video),    # Reference second
            '-filter_complex', vmaf_filter,
            '-f', 'null',
            '-'
        ]
        
        try:
            # Run FFmpeg (suppress output)
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Parse VMAF output
            if output_json.exists():
                with open(output_json, 'r') as f:
                    data = json.load(f)
                    
                # Extract mean VMAF score
                if 'pooled_metrics' in data:
                    vmaf_score = data['pooled_metrics']['vmaf']['mean']
                    return round(vmaf_score, 2)
            
            return None
            
        except Exception as e:
            print(f"    ✗ VMAF calculation failed: {e}")
            return None
    
    def calculate_for_video(
        self,
        video_name: str,
        bitrate_levels: List[int]
    ) -> Dict[int, float]:
        """
        Calculate VMAF for all bitrate levels of a video.
        
        Args:
            video_name: Name of the video
            bitrate_levels: List of bitrate levels to evaluate
            
        Returns:
            Dictionary mapping bitrate to VMAF score
        """
        reference_video = self.reference_dir / f"{video_name}.mp4"
        
        if not reference_video.exists():
            print(f"✗ Reference video not found: {reference_video}")
            return {}
        
        encoded_dir = self.encoded_dir / video_name
        if not encoded_dir.exists():
            print(f"✗ Encoded directory not found: {encoded_dir}")
            return {}
        
        # Create output directory for this video
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        vmaf_scores = {}
        
        print(f"\n📊 Calculating VMAF for: {video_name}")
        
        for idx, bitrate in enumerate(bitrate_levels, 1):
            encoded_video = encoded_dir / f"{video_name}_{bitrate}kbps.mp4"
            output_json = video_output_dir / f"vmaf_{bitrate}kbps.json"
            
            if not encoded_video.exists():
                print(f"  [{idx}/{len(bitrate_levels)}] {bitrate} Kbps - ✗ File not found")
                continue
            
            # Check if already calculated
            if output_json.exists():
                try:
                    with open(output_json, 'r') as f:
                        data = json.load(f)
                        vmaf_score = data['pooled_metrics']['vmaf']['mean']
                        vmaf_scores[bitrate] = round(vmaf_score, 2)
                        print(f"  [{idx}/{len(bitrate_levels)}] {bitrate} Kbps - "
                              f"✓ Cached (VMAF: {vmaf_scores[bitrate]:.2f})")
                        continue
                except:
                    pass
            
            print(f"  [{idx}/{len(bitrate_levels)}] {bitrate} Kbps - Calculating...", end=' ')
            
            vmaf_score = self.calculate_vmaf(
                reference_video=reference_video,
                distorted_video=encoded_video,
                output_json=output_json
            )
            
            if vmaf_score is not None:
                vmaf_scores[bitrate] = vmaf_score
                print(f"✓ VMAF: {vmaf_score:.2f}")
            else:
                print("✗ Failed")
        
        return vmaf_scores
    
    def calculate_all_videos(
        self,
        bitrate_levels: List[int] = [300, 750, 1200, 1850, 2850, 6000],
        parallel: bool = False
    ) -> pd.DataFrame:
        """
        Calculate VMAF for all videos and bitrates.
        
        Args:
            bitrate_levels: List of bitrate levels
            parallel: Use parallel processing (faster but more CPU)
            
        Returns:
            DataFrame with VMAF scores
        """
        # Find all videos
        video_dirs = [d for d in self.encoded_dir.iterdir() if d.is_dir()]
        
        if not video_dirs:
            print("✗ No encoded videos found")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print(f"VMAF Calculation Pipeline")
        print(f"Videos: {len(video_dirs)}")
        print(f"Bitrate levels: {bitrate_levels}")
        print(f"Parallel processing: {parallel}")
        print(f"{'='*60}")
        
        all_results = []
        
        if parallel:
            # Parallel processing (faster)
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        self.calculate_for_video,
                        video_dir.name,
                        bitrate_levels
                    ): video_dir.name
                    for video_dir in video_dirs
                }
                
                for future in as_completed(futures):
                    video_name = futures[future]
                    try:
                        vmaf_scores = future.result()
                        for bitrate, score in vmaf_scores.items():
                            all_results.append({
                                'video': video_name,
                                'bitrate_kbps': bitrate,
                                'vmaf': score
                            })
                    except Exception as e:
                        print(f"✗ Error processing {video_name}: {e}")
        else:
            # Sequential processing (easier to debug)
            for video_dir in video_dirs:
                video_name = video_dir.name
                vmaf_scores = self.calculate_for_video(video_name, bitrate_levels)
                
                for bitrate, score in vmaf_scores.items():
                    all_results.append({
                        'video': video_name,
                        'bitrate_kbps': bitrate,
                        'vmaf': score
                    })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            # Sort by video and bitrate
            df = df.sort_values(['video', 'bitrate_kbps']).reset_index(drop=True)
            
            # Save to CSV
            csv_path = self.output_dir / 'vmaf_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✓ VMAF summary saved to: {csv_path}")
        
        return df
    
    def get_vmaf_summary(self) -> pd.DataFrame:
        """Load or generate VMAF summary."""
        csv_path = self.output_dir / 'vmaf_summary.csv'
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            return self.calculate_all_videos()
    
    def print_summary(self, df: pd.DataFrame):
        """Print nice summary of VMAF scores."""
        if df.empty:
            print("No VMAF data available")
            return
        
        print(f"\n{'='*60}")
        print("VMAF Summary")
        print(f"{'='*60}\n")
        
        for video in df['video'].unique():
            video_df = df[df['video'] == video]
            print(f"📹 {video}:")
            
            for _, row in video_df.iterrows():
                bitrate = row['bitrate_kbps']
                vmaf = row['vmaf']
                
                # Quality indicator
                if vmaf >= 80:
                    quality = "🟢 Excellent"
                elif vmaf >= 60:
                    quality = "🟡 Good"
                elif vmaf >= 40:
                    quality = "🟠 Fair"
                else:
                    quality = "🔴 Poor"
                
                print(f"  {bitrate:4d} Kbps → VMAF: {vmaf:5.2f} {quality}")
            
            print()


def main():
    """Main VMAF calculation script."""
    print("\n📊 VMAF Calculator for ABR Research\n")
    
    calculator = VMAFCalculator(
        reference_dir='data/raw_videos',
        encoded_dir='data/encoded_videos',
        output_dir='data/vmaf_scores'
    )
    
    # Calculate VMAF for all videos
    df = calculator.calculate_all_videos(parallel=False)
    
    if not df.empty:
        calculator.print_summary(df)
        
        print("\n✓ VMAF calculation complete!")
        print("\nNext step: Extract SI/TI features")
        print("  python src/data_preparation/si_ti_extractor.py")
    else:
        print("\n✗ No VMAF scores calculated")


if __name__ == '__main__':
    main()