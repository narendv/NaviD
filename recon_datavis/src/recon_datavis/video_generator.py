import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import os
import cv2
from tqdm import tqdm

from recon_datavis.gps.plotter import GPSPlotter
from recon_datavis import pyblit
from recon_datavis.utils import bytes2im


class VideoGenerator:
    """Generate videos from HDF5 trajectory data."""
    
    def __init__(self, hdf5_fname, output_path, fps=10, figsize=(20, 10)):
        """
        Initialize video generator.
        
        Args:
            hdf5_fname: Path to HDF5 file
            output_path: Output video file path (should end with .mp4 or .avi)
            fps: Frames per second for output video
            figsize: Figure size for visualization
        """
        self.hdf5_fname = hdf5_fname
        self.output_path = output_path
        self.fps = fps
        self.figsize = figsize
        
        # Load HDF5 file
        self.hdf5 = h5py.File(hdf5_fname, 'r')
        self.hdf5_len = len(self.hdf5['collision/any'])
        
        # Setup visualization components
        self._setup_visualization()
        
        # Video writer will be initialized when we start recording
        self.video_writer = None
        
    def _setup_visualization(self):
        """Setup matplotlib visualization similar to HDF5Visualizer."""
        self.fig, axes = plt.subplots(2, 4, figsize=self.figsize)
        ((ax_rgb_left, ax_rgb_right, ax_thermal, ax_lidar),
         (self.ax_gpscompass, ax_coll, ax_imu, ax_speedsteer)) = axes
        
        # RGB Left
        self.pyblit_rgb_left = pyblit.Imshow(ax_rgb_left)
        self.pyblit_rgb_left_ax = pyblit.Axis(ax_rgb_left, [self.pyblit_rgb_left])
        ax_rgb_left.set_title('RGB Left Camera')
        
        # RGB Right
        self.pyblit_rgb_right = pyblit.Imshow(ax_rgb_right)
        self.pyblit_rgb_right_illuminance = pyblit.Text(ax_rgb_right)
        self.pyblit_rgb_right_ax = pyblit.Axis(ax_rgb_right, [self.pyblit_rgb_right, self.pyblit_rgb_right_illuminance])
        ax_rgb_right.set_title('RGB Right Camera')
        
        # Thermal
        self.pyblit_thermal = pyblit.Imshow(ax_thermal)
        self.pyblit_thermal_text = pyblit.Text(ax_thermal)
        self.pyblit_thermal_ax = pyblit.Axis(ax_thermal, [self.pyblit_thermal, self.pyblit_thermal_text])
        ax_thermal.set_title('Thermal Camera')
        
        # LiDAR
        self.pyblit_lidar = pyblit.Scatter(ax_lidar)
        self.pyblit_lidar_ax = pyblit.Axis(ax_lidar, [self.pyblit_lidar])
        ax_lidar.set_title('LiDAR')
        
        # GPS/Compass (simplified without Google Maps)
        self.ax_gpscompass.set_title('GPS/Compass')
        
        # Collision
        self.pyblit_coll = pyblit.Bar(ax_coll)
        self.pyblit_coll_ax = pyblit.Axis(ax_coll, [self.pyblit_coll])
        
        # IMU
        self.pyblit_imu = pyblit.Bar(ax_imu)
        self.pyblit_imu_jackal = pyblit.Bar(ax_imu)
        self.pyblit_imu_ax = pyblit.Axis(ax_imu, [self.pyblit_imu, self.pyblit_imu_jackal])
        
        # Speed/Steer
        self.pyblit_speedsteer_commanded_linvel = pyblit.Line(ax_speedsteer)
        self.pyblit_speedsteer_commanded_angvel = pyblit.Line(ax_speedsteer)
        self.pyblit_speedsteer_actual_linvel = pyblit.Line(ax_speedsteer)
        self.pyblit_speedsteer_actual_angvel = pyblit.Line(ax_speedsteer)
        self.pyblit_speedsteer_legend = pyblit.Legend(ax_speedsteer)
        self.pyblit_speedsteer_ax = pyblit.Axis(ax_speedsteer,
                                               [self.pyblit_speedsteer_commanded_linvel,
                                                self.pyblit_speedsteer_commanded_angvel,
                                                self.pyblit_speedsteer_actual_linvel,
                                                self.pyblit_speedsteer_actual_angvel,
                                                self.pyblit_speedsteer_legend])
        
        plt.tight_layout()
        
    def _get_hdf5_topic(self, topic, timestep):
        """Get data from HDF5 topic at specific timestep."""
        value = self.hdf5[topic][timestep]
        if type(value) == np.string_:
            value = bytes2im(value)
        return value
    
    def _plot_lidar(self, timestep):
        """Plot LiDAR data."""
        lidar = self._get_hdf5_topic('lidar', timestep)
        assert len(lidar) == 360

        measurement_indices = list(range(300, 360)) + list(range(0, 60))
        angles = np.deg2rad(np.linspace(30, 150., 120))
        lidar = lidar[measurement_indices]
        lidar_notfinite = np.logical_not(np.isfinite(lidar))
        max_range = 15.
        lidar[lidar_notfinite] = max_range

        x = lidar * np.cos(angles)
        y = lidar * np.sin(angles)
        c = [to_rgb('r') if notfinite else to_rgb('k') for notfinite in lidar_notfinite]

        self.pyblit_lidar.draw(x, y, c=c, s=2.)
        self.pyblit_lidar_ax.ax.set_xlim((-16, 16))
        self.pyblit_lidar_ax.ax.set_ylim((0, 16))
        self.pyblit_lidar_ax.ax.set_aspect('equal')
        self.pyblit_lidar_ax.draw()
    
    def _plot_speedsteer(self, timestep):
        """Plot speed/steer commands."""
        commanded_linvel = self._get_hdf5_topic('commands/linear_velocity', timestep)
        commanded_angvel = -self._get_hdf5_topic('commands/angular_velocity', timestep)
        actual_linvel = self._get_hdf5_topic('jackal/linear_velocity', timestep)
        actual_angvel = -self._get_hdf5_topic('jackal/angular_velocity', timestep)

        self.pyblit_speedsteer_commanded_linvel.draw([0, 0], [0, commanded_linvel],
                                                    linestyle='-', color='k', linewidth=10.0, label='Commanded')
        self.pyblit_speedsteer_commanded_angvel.draw([0, commanded_angvel], [0, 0],
                                                    linestyle='-', color='k', linewidth=10.0)
        self.pyblit_speedsteer_actual_linvel.draw([0, 0], [0, actual_linvel],
                                                 linestyle='-', color='c', linewidth=4.0, label='Actual')
        self.pyblit_speedsteer_actual_angvel.draw([0, actual_angvel], [0, 0],
                                                 linestyle='-', color='c', linewidth=4.0)
        self.pyblit_speedsteer_legend.draw(loc='lower left')
        ax = self.pyblit_speedsteer_ax.ax
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_title('Speed / Steer')
        self.pyblit_speedsteer_ax.draw()
    
    def _plot_collision(self, timestep):
        """Plot collision data."""
        topics = ['collision/{0}'.format(name) for name in
                  ('any', 'physical', 'close', 'flipped', 'stuck', 'outside_geofence')]
        colls = [self._get_hdf5_topic(topic, timestep) for topic in topics]

        self.pyblit_coll.draw(
            np.arange(len(colls)),
            colls,
            tick_label=[topic.replace('collision/', '').replace('outside_', '') for topic in topics],
            color='r'
        )
        ax = self.pyblit_coll_ax.ax
        ax.set_title('Collision')
        ax.set_xlim((-0.5, len(colls) + 0.5))
        ax.set_ylim((0, 1))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        self.pyblit_coll_ax.draw()
    
    def _plot_imu(self, timestep):
        """Plot IMU data."""
        linacc = self._get_hdf5_topic('imu/linear_acceleration', timestep)
        angvel = self._get_hdf5_topic('imu/angular_velocity', timestep)
        jackal_linacc = self._get_hdf5_topic('jackal/imu/linear_acceleration', timestep)
        jackal_angvel = self._get_hdf5_topic('jackal/imu/angular_velocity', timestep)

        linacc[2] = -linacc[2] + 9.81  # negative since upside down, and subtract off gravity offset
        jackal_linacc[2] -= 9.81

        self.pyblit_imu.draw(
            np.arange(6),
            np.concatenate((linacc, angvel)),
            tick_label=['linacc', '', '', 'angvel', '', ''],
            color='k',
            width=0.8
        )
        self.pyblit_imu_jackal.draw(
            np.arange(6),
            np.concatenate((jackal_linacc, jackal_angvel)),
            tick_label=['linacc', '', '', 'angvel', '', ''],
            color='r',
            width=0.5
        )
        ax = self.pyblit_imu_ax.ax
        ax.set_title('IMU')
        ax.set_xlim((-0.5, 6.5))
        ax.set_ylim((-3., 3.))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        self.pyblit_imu_ax.draw()
    
    def _plot_gpscompass(self, timestep):
        """Plot GPS/compass data (simplified)."""
        compass_bearing = self._get_hdf5_topic('imu/compass_bearing', timestep)
        latlong = self._get_hdf5_topic('gps/latlong', timestep)
        
        # Clear the axis
        self.ax_gpscompass.clear()
        self.ax_gpscompass.set_title('GPS/Compass')
        
        # Plot a simple compass rose
        self.ax_gpscompass.set_xlim(-1, 1)
        self.ax_gpscompass.set_ylim(-1, 1)
        self.ax_gpscompass.set_aspect('equal')
        
        # Draw compass circle
        circle = plt.Circle((0, 0), 0.9, fill=False, color='black')
        self.ax_gpscompass.add_patch(circle)
        
        # Draw compass bearing arrow
        if np.isfinite(compass_bearing):
            bearing_rad = np.deg2rad(compass_bearing)
            dx = 0.8 * np.sin(bearing_rad)
            dy = 0.8 * np.cos(bearing_rad)
            self.ax_gpscompass.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Add GPS coordinates as text
        if np.isfinite(latlong).all():
            self.ax_gpscompass.text(0, -1.2, f'GPS: {latlong[0]:.6f}, {latlong[1]:.6f}', 
                                   ha='center', va='top', fontsize=8)
        else:
            self.ax_gpscompass.text(0, -1.2, 'GPS: No Fix', ha='center', va='top', fontsize=8)
        
        # Add compass bearing text
        self.ax_gpscompass.text(0, 1.2, f'Bearing: {compass_bearing:.1f}°', 
                               ha='center', va='bottom', fontsize=8)
    
    def _update_visualization(self, timestep):
        """Update all visualization components for given timestep."""
        # Update RGB cameras
        self.pyblit_rgb_left.draw(self._get_hdf5_topic('images/rgb_left', timestep))
        self.pyblit_rgb_left_ax.draw()

        self.pyblit_rgb_right.draw(self._get_hdf5_topic('images/rgb_right', timestep))
        self.pyblit_rgb_right_illuminance.draw(
            0.25, 0.0,
            text='Illuminance: {0:.0f}'.format(self._get_hdf5_topic('android/illuminance', timestep)),
            transform=self.pyblit_rgb_right_ax.ax.transAxes, fontsize=11, color='w')
        self.pyblit_rgb_right_ax.draw()

        # Update thermal camera
        self.pyblit_thermal.draw(self._get_hdf5_topic('images/thermal', timestep))
        self.pyblit_thermal_text.draw(
            0.05, 0.9,
            text=f'{os.path.basename(self.hdf5_fname)}\nFrame {timestep + 1}/{self.hdf5_len}',
            transform=self.pyblit_thermal_ax.ax.transAxes, fontsize=11, color='w')
        self.pyblit_thermal_ax.draw()

        # Update other sensors
        self._plot_lidar(timestep)
        self._plot_speedsteer(timestep)
        self._plot_collision(timestep)
        self._plot_imu(timestep)
        self._plot_gpscompass(timestep)
    
    def _initialize_video_writer(self):
        """Initialize the video writer."""
        # Get figure dimensions
        self.fig.canvas.draw()
        # Use the updated matplotlib API
        buf = self.fig.canvas.buffer_rgba()
        buf = np.asarray(buf)
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        height, width = buf.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to initialize video writer for {self.output_path}")
        
        print(f"Initialized video writer: {width}x{height} @ {self.fps}fps")
        return width, height
    
    def _capture_frame(self):
        """Capture current matplotlib figure as video frame."""
        self.fig.canvas.draw()
        # Use the updated matplotlib API
        buf = self.fig.canvas.buffer_rgba()
        buf = np.asarray(buf)
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        return frame
    
    def generate_video(self, start_frame=0, end_frame=None, step=1):
        """
        Generate video from trajectory data.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None for full trajectory)
            step: Step size (1 for every frame, 2 for every other frame, etc.)
        """
        if end_frame is None:
            end_frame = self.hdf5_len
            
        end_frame = min(end_frame, self.hdf5_len)
        
        print(f"Generating video for frames {start_frame} to {end_frame} (step={step})")
        print(f"Output: {self.output_path}")
        
        # Initialize video writer on first frame
        self._update_visualization(start_frame)
        width, height = self._initialize_video_writer()
        
        # Generate frames
        frame_indices = range(start_frame, end_frame, step)
        for timestep in tqdm(frame_indices, desc="Generating video"):
            # Update visualization
            self._update_visualization(timestep)
            
            # Capture frame
            frame = self._capture_frame()
            
            # Write frame
            self.video_writer.write(frame)
        
        # Cleanup
        self.video_writer.release()
        self.hdf5.close()
        plt.close(self.fig)
        
        print(f"Video saved to: {self.output_path}")
    
    def generate_clip(self, start_time_sec, duration_sec, step=1):
        """
        Generate video clip for a specific time range.
        
        Args:
            start_time_sec: Start time in seconds (assumes data at 10Hz)
            duration_sec: Duration in seconds
            step: Frame step size
        """
        # Assume data is collected at approximately 10Hz
        data_hz = 10
        start_frame = int(start_time_sec * data_hz)
        end_frame = int((start_time_sec + duration_sec) * data_hz)
        
        # Ensure we don't exceed trajectory bounds
        start_frame = min(start_frame, self.hdf5_len - 1)
        end_frame = min(end_frame, self.hdf5_len)
        
        if start_frame >= end_frame:
            raise ValueError(f"Invalid time range: start_frame={start_frame}, end_frame={end_frame}, trajectory_length={self.hdf5_len}")
        
        self.generate_video(start_frame, end_frame, step)