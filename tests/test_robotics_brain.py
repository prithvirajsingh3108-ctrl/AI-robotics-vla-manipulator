import pytest
import math
from robotics.kinematics import RoboticArmKinematics
from vision.yolo_detector import YoloTabletopTracker

def test_inverse_kinematics_solves_valid_xyz():
    """Validates that the analytical Inverse Kinematics solver handles realistic worktable values."""
    kinematics = RoboticArmKinematics()
    
    # Target location within high reach range (80x, 150y, 10z)
    angles = kinematics.solve_inverse_kinematics(80.0, 150.0, 10.0)
    
    assert "theta1_base_yaw" in angles
    assert "theta2_shoulder_pitch" in angles
    assert "theta3_elbow_flexion" in angles
    
    # Base Yaw Theta1 should be arctan(150 / 80) = ~61.9 degrees
    assert math.isclose(angles["theta1_base_yaw"], 61.93, abs_tol=0.5)

def test_inverse_kinematics_boundary_error():
    """Verifies that coordinates beyond links ranges raise an evaluation error."""
    kinematics = RoboticArmKinematics()
    
    # Far-reaches target coordinate that exceeds kinematics limits (L2+L3 = 190mm)
    with pytest.raises(ValueError):
        kinematics.solve_inverse_kinematics(250.0, 250.0, 100.0)

def test_forward_kinematics_reconstruction():
    """Verifies forward kinematics can accurately reconstruct coordinates from angles."""
    kinematics = RoboticArmKinematics()
    
    x, y, z = 100.0, 120.0, 30.0
    angles = kinematics.solve_inverse_kinematics(x, y, z)
    
    fk_reconstruct = kinematics.solve_forward_kinematics(
        angles["theta1_base_yaw"],
        angles["theta2_shoulder_pitch"],
        angles["theta3_elbow_flexion"]
    )
    
    assert math.isclose(fk_reconstruct["x"], x, abs_tol=1.0)
    assert math.isclose(fk_reconstruct["y"], y, abs_tol=1.0)
    assert math.isclose(fk_reconstruct["z"], z, abs_tol=1.0)

def test_centroid_tracker_mock_processing():
    """Verifies the mock visual frame generator accurately returns centroid metrics data."""
    tracker = YoloTabletopTracker()
    frame, detected = tracker.process_mock_frame()
    
    # Captured frame should have standard height 480 and width 640
    assert frame.shape == (480, 640, 3)
    assert len(detected) == 3
    
    names = [d["name"] for d in detected]
    assert "red cube" in names
    assert "blue box" in names
    assert "green sphere" in names
