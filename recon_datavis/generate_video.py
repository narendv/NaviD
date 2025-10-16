#!/usr/bin/env python3
"""
Generate videos from HDF5 trajectory data.

Usage examples:
    # Generate full trajectory video
    python generate_video.py -input /path/to/trajectory.hdf5 -output video.mp4
    
    # Generate video for specific frame range
    python generate_video.py -input /path/to/trajectory.hdf5 -output video.mp4 --start_frame 100 --end_frame 500
    
    # Generate video clip for specific time range
    python generate_video.py -input /path/to/trajectory.hdf5 -output video.mp4 --start_time 10 --duration 30
    
    # Generate video with custom settings
    python generate_video.py -input /path/to/trajectory.hdf5 -output video.mp4 --fps 15 --step 2 --figsize 24 12
"""

import argparse
import os
import sys

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recon_datavis.video_generator import VideoGenerator
from recon_datavis.utils import get_files_ending_with


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos from HDF5 trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument('-input', '--input', type=str, required=True,
                        help='Path to HDF5 trajectory file or folder containing HDF5 files')
    parser.add_argument('-output', '--output', type=str, required=True,
                        help='Output video file path (e.g., trajectory.mp4)')
    
    # Video settings
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for output video (default: 10)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[20, 10],
                        help='Figure size for visualization (width height, default: 20 10)')
    
    # Frame selection
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument('--start_frame', type=int, default=0,
                            help='Starting frame index (default: 0)')
    frame_group.add_argument('--start_time', type=float,
                            help='Starting time in seconds (assumes 10Hz data)')
    
    parser.add_argument('--end_frame', type=int,
                        help='Ending frame index (default: end of trajectory)')
    parser.add_argument('--duration', type=float,
                        help='Duration in seconds (use with --start_time)')
    parser.add_argument('--step', type=int, default=1,
                        help='Frame step size (1=every frame, 2=every other frame, default: 1)')
    
    # Multiple files
    parser.add_argument('--batch', action='store_true',
                        help='Process all HDF5 files in input folder')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_time is not None and args.duration is None:
        parser.error("--duration is required when using --start_time")
    
    if args.duration is not None and args.start_time is None:
        parser.error("--start_time is required when using --duration")
    
    # Get input files
    if os.path.isfile(args.input):
        hdf5_files = [args.input]
    elif os.path.isdir(args.input):
        if args.batch:
            hdf5_files = get_files_ending_with(args.input, '.hdf5')
            if not hdf5_files:
                print(f"No HDF5 files found in {args.input}")
                return
        else:
            print(f"Input is a directory. Use --batch to process all HDF5 files, or specify a single file.")
            return
    else:
        print(f"Input path does not exist: {args.input}")
        return
    
    # Process files
    for i, hdf5_file in enumerate(hdf5_files):
        print(f"\nProcessing file {i+1}/{len(hdf5_files)}: {os.path.basename(hdf5_file)}")
        
        # Determine output path for multiple files
        if len(hdf5_files) > 1:
            base_name = os.path.splitext(os.path.basename(hdf5_file))[0]
            output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
            output_ext = os.path.splitext(args.output)[1] or '.mp4'
            current_output = os.path.join(output_dir, f"{base_name}_video{output_ext}")
        else:
            current_output = args.output
        
        try:
            # Create video generator
            generator = VideoGenerator(
                hdf5_file, 
                current_output, 
                fps=args.fps,
                figsize=tuple(args.figsize)
            )
            
            # Generate video based on arguments
            if args.start_time is not None:
                generator.generate_clip(args.start_time, args.duration, args.step)
            else:
                generator.generate_video(
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    step=args.step
                )
            
            print(f"✅ Video saved: {current_output}")
            
        except Exception as e:
            print(f"❌ Error processing {hdf5_file}: {e}")
            continue


if __name__ == "__main__":
    main()