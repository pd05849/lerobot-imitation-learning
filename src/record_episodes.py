"""
Record teleoperation episodes for SO-101 robot
Captures synchronized camera feeds and joint configurations
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from lerobot import Robot, Camera, DatasetRecorder

def setup_cameras(camera_ids: list = [0, 1]):
    """
    Initialize cameras for recording
    
    Args:
        camera_ids: List of camera device IDs [overhead, laptop]
    
    Returns:
        Dictionary of Camera objects
    """
    cameras = {}
    camera_names = ["overhead", "laptop"]
    
    for idx, (cam_id, cam_name) in enumerate(zip(camera_ids, camera_names)):
        print(f"Initializing {cam_name} camera (ID: {cam_id})...")
        cameras[cam_name] = Camera(
            device_id=cam_id,
            fps=30,
            width=640,
            height=480
        )
    
    return cameras

def record_episode(
    robot: Robot,
    cameras: dict,
    episode_num: int,
    output_dir: Path,
    episode_duration: float = 10.0,
    warmup_time: float = 3.0
):
    """
    Record a single teleoperation episode
    
    Args:
        robot: Robot instance
        cameras: Dictionary of Camera objects
        episode_num: Episode number
        output_dir: Directory to save episode data
        episode_duration: Duration of recording in seconds
        warmup_time: Preparation time before recording starts
    """
    print(f"\n=== Episode {episode_num} ===")
    print(f"Preparation time: {warmup_time}s")
    print("Position robot at starting configuration...")
    
    # Countdown
    for i in range(int(warmup_time), 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print("Recording!")
    
    # Initialize recorder
    recorder = DatasetRecorder(
        output_dir=output_dir,
        episode_num=episode_num,
        fps=30
    )
    
    start_time = time.time()
    frame_num = 0
    
    try:
        while time.time() - start_time < episode_duration:
            # Capture camera frames
            frames = {}
            for cam_name, camera in cameras.items():
                frames[f"camera_{cam_name}"] = camera.read()
            
            # Get robot state
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()
            gripper_state = robot.get_gripper_state()
            
            # Record frame
            recorder.add_frame(
                timestamp=time.time() - start_time,
                observations={
                    **frames,
                    "joint_positions": joint_positions,
                    "joint_velocities": joint_velocities,
                    "gripper_state": gripper_state
                },
                actions={
                    "target_joint_positions": joint_positions,  # Current pos as target
                    "gripper_command": gripper_state
                }
            )
            
            frame_num += 1
            
            # Display progress
            elapsed = time.time() - start_time
            progress = (elapsed / episode_duration) * 100
            print(f"\rProgress: {progress:.1f}% | Frame: {frame_num}", end="", flush=True)
        
        print(f"\n✓ Episode {episode_num} recorded ({frame_num} frames)")
        
    finally:
        # Save episode
        recorder.save()
        print(f"Episode saved to {output_dir}/episode_{episode_num:03d}.hdf5")

def main():
    parser = argparse.ArgumentParser(description="Record SO-101 teleoperation episodes")
    parser.add_argument("--robot-path", type=str, default="so101",
                      help="Path to robot configuration")
    parser.add_argument("--output-dir", type=str, default="./data",
                      help="Output directory for dataset")
    parser.add_argument("--num-episodes", type=int, default=50,
                      help="Number of episodes to record")
    parser.add_argument("--episode-duration", type=float, default=10.0,
                      help="Duration of each episode (seconds)")
    parser.add_argument("--warmup-time", type=float, default=3.0,
                      help="Preparation time before each episode (seconds)")
    parser.add_argument("--reset-time", type=float, default=5.0,
                      help="Reset time between episodes (seconds)")
    parser.add_argument("--camera-ids", type=int, nargs=2, default=[0, 1],
                      help="Camera device IDs [overhead, laptop]")
    parser.add_argument("--start-episode", type=int, default=0,
                      help="Starting episode number")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== SO-101 Episode Recording ===")
    print(f"Output directory: {output_dir}")
    print(f"Episodes to record: {args.num_episodes}")
    print(f"Episode duration: {args.episode_duration}s")
    
    # Initialize hardware
    print("\nInitializing robot...")
    robot = Robot(args.robot_path)
    
    print("\nInitializing cameras...")
    cameras = setup_cameras(args.camera_ids)
    
    # Test cameras
    print("\nTesting cameras...")
    for cam_name, camera in cameras.items():
        frame = camera.read()
        print(f"  {cam_name}: {frame.shape} at {camera.fps} fps")
    
    print("\n=== Recording Instructions ===")
    print("1. Ensure robot starts and ends at zero configuration")
    print("2. Complete the task within the episode duration")
    print("3. If episode fails, you can re-record it later")
    print("\nPress Enter to start recording...")
    input()
    
    # Record episodes
    try:
        for ep_num in range(args.start_episode, args.start_episode + args.num_episodes):
            # Record episode
            record_episode(
                robot=robot,
                cameras=cameras,
                episode_num=ep_num,
                output_dir=output_dir,
                episode_duration=args.episode_duration,
                warmup_time=args.warmup_time
            )
            
            # Reset between episodes
            if ep_num < args.start_episode + args.num_episodes - 1:
                print(f"\nReset robot to starting position ({args.reset_time}s)")
                time.sleep(args.reset_time)
    
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        robot.disconnect()
        for camera in cameras.values():
            camera.release()
        
        print(f"\n=== Recording Complete ===")
        print(f"Total episodes recorded: {ep_num - args.start_episode + 1}")
        print(f"Dataset location: {output_dir}")

if __name__ == "__main__":
    main()
