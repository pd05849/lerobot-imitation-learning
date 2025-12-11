"""
Calibrate SO-101 Robot Arm
Establishes encoder-angle relationships, home positions, and torque limits
"""

import argparse
import time
from lerobot import Robot

def calibrate_motors(robot_path: str = "so101", port: str = None):
    """
    Calibrate all servo motors in the SO-101 arm
    
    Args:
        robot_path: Path to robot configuration
        port: Serial port for motor communication (auto-detect if None)
    """
    print("Initializing robot connection...")
    robot = Robot(robot_path, port=port)
    
    print("\n=== Motor Detection ===")
    motors = robot.detect_motors()
    print(f"Found {len(motors)} motors")
    for motor_id, motor_info in motors.items():
        print(f"Motor {motor_id}: {motor_info}")
    
    print("\n=== Calibration Process ===")
    print("Please manually move each joint through its full range of motion")
    print("Press Enter when ready to start...")
    input()
    
    # Calibrate each motor
    for motor_id in motors.keys():
        print(f"\nCalibrating Motor {motor_id}...")
        print("Move joint to MINIMUM position, then press Enter")
        input()
        min_pos = robot.get_motor_position(motor_id)
        
        print("Move joint to MAXIMUM position, then press Enter")
        input()
        max_pos = robot.get_motor_position(motor_id)
        
        print("Move joint to HOME (zero) position, then press Enter")
        input()
        home_pos = robot.get_motor_position(motor_id)
        
        # Save calibration
        robot.set_calibration(motor_id, min_pos, max_pos, home_pos)
        print(f"Motor {motor_id} calibrated:")
        print(f"  Min: {min_pos:.2f}, Max: {max_pos:.2f}, Home: {home_pos:.2f}")
    
    print("\n=== Setting Torque Limits ===")
    # Set safe torque limits (adjust based on your hardware)
    torque_limit = 0.8  # 80% of maximum torque
    for motor_id in motors.keys():
        robot.set_torque_limit(motor_id, torque_limit)
        print(f"Motor {motor_id}: Torque limit set to {torque_limit*100}%")
    
    print("\n=== Calibration Complete ===")
    print("Calibration parameters saved to robot configuration")
    
    # Test calibration
    print("\nTesting calibration - moving to home position...")
    robot.move_to_home()
    time.sleep(2)
    
    print("Calibration successful!")
    robot.disconnect()

def verify_calibration(robot_path: str = "so101"):
    """Verify calibration by checking motor responses"""
    print("Verifying calibration...")
    robot = Robot(robot_path)
    
    # Check each motor can reach home position
    print("Moving to home position...")
    robot.move_to_home()
    time.sleep(2)
    
    current_pos = robot.get_joint_positions()
    print(f"Current joint positions: {current_pos}")
    
    # Check if positions are close to zero (within tolerance)
    tolerance = 0.1  # radians
    all_good = all(abs(pos) < tolerance for pos in current_pos)
    
    if all_good:
        print("✓ Calibration verified successfully")
    else:
        print("✗ Calibration verification failed - positions not at home")
        print("  Please re-run calibration")
    
    robot.disconnect()
    return all_good

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate SO-101 robot arm")
    parser.add_argument("--robot-path", type=str, default="so101",
                      help="Path to robot configuration")
    parser.add_argument("--port", type=str, default=None,
                      help="Serial port for motors (auto-detect if not specified)")
    parser.add_argument("--verify-only", action="store_true",
                      help="Only verify existing calibration")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_calibration(args.robot_path)
    else:
        calibrate_motors(args.robot_path, args.port)
        verify_calibration(args.robot_path)
