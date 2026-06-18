import math

class RoboticArmKinematics:
    def __init__(self, l1=80.0, l2=100.0, l3=90.0):
        """
        Maintains the fixed links characteristics of our 3-DoF Desktop Robotic Arm manipulator.
        
        Args:
            l1 (float): Height from table base center to shoulder joint axis in millimeters.
            l2 (float): Span of upper shoulder-to-elbow arm segment in millimeters.
            l3 (float): Span of elbow-to-wrist gripper tip segment in millimeters.
        """
        self.L1 = l1
        self.L2 = l2
        self.L3 = l3

    def solve_inverse_kinematics(self, x: float, y: float, z: float):
        """
        Computes rotational joint angles required to position end-effector tip at (X, Y, Z).
        
        Formula uses analytical trigonometry equations. Returns angle configurations in degrees.
        """
        # 1. Calculate Base Joint Axis (Yaw rotation on XY Plane)
        theta1_rad = math.atan2(y, x)
        theta1_deg = math.degrees(theta1_rad)

        # 2. Project 3D vector length into planar 2D vertical frame (Radial Projection R)
        r = math.sqrt(x**2 + y**2)
        
        # Offset destination coordinate relative to shoulder anchor height position (Z relative)
        z_rel = z - self.L1

        # Direct diagonal hypotenuse span from shoulder joint centerline axis to tip coordinates
        d = math.sqrt(r**2 + z_rel**2)
        if d > (self.L2 + self.self.L3 if hasattr(self, 'self') else self.L2 + self.L3):
            raise ValueError(f"Goal position ({x}, {y}, {z}) is out of physical envelope bounds.")

        # 3. Apply Law of Cosines to solve Elbow flexion configuration angle (Theta 3)
        cos_theta3 = (d**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_theta3 = max(-1.0, min(1.0, cos_theta3))  # Clamp value to correct rounding margins
        
        theta3_rad = math.acos(cos_theta3)
        theta3_deg = math.degrees(theta3_rad) - 90.0  # Offset coordinate so zero position is upright straight link

        # 4. Solves for Shoulder joint pitch angle target (Theta 2)
        alpha = math.atan2(z_rel, r)
        
        beta_num = self.L3 * math.sin(theta3_rad)
        beta_den = self.L2 + self.L3 * math.cos(theta3_rad)
        beta = math.atan2(beta_num, beta_den)
        
        theta2_deg = math.degrees(alpha + beta)

        return {
            "theta1_base_yaw": round(theta1_deg, 2),
            "theta2_shoulder_pitch": round(theta2_deg, 2),
            "theta3_elbow_flexion": round(theta3_deg, 2)
        }

    def solve_forward_kinematics(self, t1_deg: float, t2_deg: float, t3_deg: float):
        """
        Computes Cartesian positions of end-effector claw based on rotational servo positions.
        """
        t1 = math.radians(t1_deg)
        t2 = math.radians(t2_deg)
        t3 = math.radians(t3_deg + 90.0)  # Revert normal mathematical coordinate alignment offset
        
        # Determine Radial Reach projection from joints
        r = self.L2 * math.cos(t2) + self.L3 * math.cos(t2 + t3)
        
        # Transform 2D radial offset back to 3D workspace frames
        x = r * math.cos(t1)
        y = r * math.sin(t1)
        z = self.L1 + self.L2 * math.sin(t2) + self.L3 * math.sin(t2 + t3)
        
        return {
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2)
        }
