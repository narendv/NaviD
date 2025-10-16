#!/usr/bin/env python3
"""
Simple example script to generate a video from an HDF5 trajectory file.
"""

import os
import sys

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recon_datavis.video_generator import VideoGenerator


def generate_example_video():
    """Generate an example video from the first HDF5 file found."""
    
    # Find first HDF5 file in the recon data directory
    data_dir = "/data/naren/recon_release"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir path in this script.")
        return
    
    # Find HDF5 files
    hdf5_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return
    
    # Use first file
    hdf5_file = hdf5_files[0]
    output_file = "traj_videos/example_trajectory.mp4"
    
    print(f"Generating video from: {hdf5_file}")
    print(f"Output: {output_file}")
    
    try:
        # Create video generator
        generator = VideoGenerator(
            hdf5_fname=hdf5_file,
            output_path=output_file,
            fps=15,  # 15 FPS for smoother video
            figsize=(24, 12)  # Larger figure for better quality
        )
        
        print(f"Trajectory length: {generator.hdf5_len} frames")
        
        # Determine what to generate based on trajectory length
        if generator.hdf5_len < 100:
            # Short trajectory - generate the entire thing
            print(f"Short trajectory ({generator.hdf5_len} frames), generating full video...")
            generator.generate_video(start_frame=0, end_frame=None, step=1)
        elif generator.hdf5_len < 400:
            # Medium trajectory - generate from start
            duration_frames = min(300, generator.hdf5_len - 10)  # Up to 30 seconds at 10Hz
            print(f"Generating first {duration_frames} frames...")
            generator.generate_video(start_frame=10, end_frame=10 + duration_frames, step=1)
        else:
            # Long trajectory - generate a 30-second clip starting from 10 seconds
            print("Generating 30-second clip starting from 10 seconds...")
            generator.generate_clip(start_time_sec=10, duration_sec=30, step=1)
        
        print(f"✅ Video generated successfully: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    generate_example_video()