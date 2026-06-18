import logging
import uuid
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field

# Core modules imports
from planner.command_parser import CommandParser, ParsedCommand
from planner.world_model import WorldModel, WorkspaceObject
from planner.task_planner import TaskPlanner
from vision.camera import TabletopCamera
from vision.yolo_detector import YoloObjectDetector
from robotics.kinematics import RoboticArmKinematics
from memory.db_session import get_db_connection

# Configure engine logger
logger = logging.getLogger("ReasoningEngine")


class AutonomousPipelineResult(BaseModel):
    task_id: str = Field(..., description="Unique generated task ID")
    raw_command: str = Field(..., description="Input query from the user")
    source: str = Field(..., description="Identified source coordinate item")
    target: str = Field(..., description="Identified target reference item")
    actions: List[str] = Field(..., description="Determined discrete action list")
    detected_objects: List[dict] = Field(..., description="Perceived items during camera execution")
    trajectory_steps: List[dict] = Field(..., description="Cartesian trajectories with physical joints rotation degrees")
    success: bool = Field(..., description="Indication if the entire mechanical sequence passed reachability and safety checks")
    message: str = Field(..., description="Informative diagnostics log")


class AutonomousReasoningEngine:
    def __init__(self):
        """
        Orchestration controller coordinating:
          1. Perception (Webcam + Detector)
          2. Command Parsing (NLP parser terms)
          3. Physical State (WorldModel registry)
          4. Inverse Kinematics (Servo Degree constraints)
          5. Persistent Memory Storage (SQLite transactional history logs)
        """
        self.camera = TabletopCamera()
        self.detector = YoloObjectDetector()
        self.parser = CommandParser()
        self.world_model = WorldModel()
        self.kinematics = RoboticArmKinematics()
        self.planner = TaskPlanner(parser=self.parser, world=self.world_model)

    def run_pipeline(self, raw_command: str) -> Dict[str, Any]:
        """
        Executes the complete autonomous reasoning loop:
        Perceive -> Parse -> Register State -> Formulate Path Trajectory -> Kinematics Validate -> Write Persistent Records.
        """
        logger.info(f"Initiating autonomous reasoning engine for operator query: '{raw_command}'")
        task_id = f"TASK-{str(uuid.uuid4())[:8].upper()}"
        success = True
        diagnostics_msgs = []

        # ----------------- STAGE 1: Object Detection / Perception -----------------
        frame, is_simulated = self.camera.read_frame()
        perceived_items, is_human_present = self.detector.detect_and_track_objects(frame)
        logger.info(f"Perception complete. Found {len(perceived_items)} items on workbench (Simulated Frame: {is_simulated}). Human detected: {is_human_present}")

        if is_human_present:
            success = False
            diagnostics_msgs.append("Safety lock active: Human intruder / hand detected inside physical workspace. Halting robotic actions.")

        # ----------------- STAGE 2: World Model State Synch -----------------
        # Dynamic mapped vocabulary to standard database keys:
        display_name_map = {
            "Red Cube": "red_cube",
            "Blue Box": "blue_box",
            "Green Sphere": "green_sphere",
            "Yellow Container": "yellow_container",
            "Orange Pyramid": "orange_pyramid"
        }

        # Clear and update live spatial world model based on fresh video measurements
        for item in perceived_items:
            display_name = item["object_name"]
            normalized_name = display_name_map.get(display_name, display_name.lower().replace(" ", "_"))
            
            # Map shape estimations
            shape = "cube"
            if "box" in normalized_name or "container" in normalized_name:
                shape = "box"
            elif "sphere" in normalized_name:
                shape = "sphere"
            elif "pyramid" in normalized_name:
                shape = "pyramid"

            self.world_model.register_object(WorkspaceObject(
                name=normalized_name,
                shape=shape,
                color_hex="#ef4444" if "red" in normalized_name else "#3b82f6" if "blue" in normalized_name else "#22c55e" if "green" in normalized_name else "#eab308" if "yellow" in normalized_name else "#f97316",
                x=item["x"],
                y=item["y"],
                z=10.0 if shape == "cube" or shape == "sphere" else 20.0,
                radius_or_extent=15.0 if shape == "cube" else 25.0
            ))

            # Push visual update into persistent SQLite database as well
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                INSERT INTO workspace_objects (item_name, shape, color_hex, loc_x, loc_y, loc_z)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_name) DO UPDATE SET loc_x=excluded.loc_x, loc_y=excluded.loc_y, loc_z=excluded.loc_z, updated_at=CURRENT_TIMESTAMP;
                """, (
                    normalized_name.replace("_", " "),
                    shape,
                    "#ef4444" if "red" in normalized_name else "#3b82f6" if "blue" in normalized_name else "#22c55e",
                    item["x"],
                    item["y"],
                    10.0
                ))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Error persisting workspace objects state update: {e}")
            finally:
                conn.close()

        # ----------------- STAGE 3: Command Parsing & Task Formulation -----------------
        parsed_cmd: ParsedCommand = self.parser.parse_instruction(raw_command, self.world_model)
        source_name = parsed_cmd.source
        target_name = parsed_cmd.target

        logger.info(f"Parsed Command. Action Plan: Manipulate source={source_name} into target={target_name}.")

        # Retrieve source/target coordinates from our fresh World Model
        source_obj = self.world_model.get_object(source_name)
        target_obj = self.world_model.get_object(target_name)

        if not source_obj:
            logger.warning(f"Could not localize target product '{source_name}', registering physical defaults.")
            # Default lookup fallback configuration
            self.world_model.register_object(WorkspaceObject(
                name=source_name, shape="cube", color_hex="#ef4444", x=80.0, y=150.0, z=10.0
            ))
            source_obj = self.world_model.get_object(source_name)

        if not target_obj:
            logger.warning(f"Could not localize receptor '{target_name}', registering standard container targets.")
            self.world_model.register_object(WorkspaceObject(
                name=target_name, shape="box", color_hex="#3b82f6", x=-100.0, y=180.0, z=20.0
            ))
            target_obj = self.world_model.get_object(target_name)

        # ----------------- STAGE 4: Task Trail Path & Kinematics Validation -----------------
        # Delegate coordinate trace trajectory formulation to TaskPlanner
        raw_plan = self.planner.generate_task_plan(raw_command)
        plan_steps = raw_plan["meta_telemetry"]["step_detail_logs"]
        
        trajectory_steps_processed = []

        for step in plan_steps:
            step_idx = step["step_index"]
            act_type = step["action_type"]
            coord_target = step["target_coordinates"]
            desc = step["description"]

            # Solve Inverse Kinematics targets
            try:
                joint_angles = self.kinematics.solve_inverse_kinematics(
                    coord_target["x"], coord_target["y"], coord_target["z"]
                )
                reach_error = None
            except ValueError as ve:
                joint_angles = {"theta1_base_yaw": 0.0, "theta2_shoulder_pitch": 0.0, "theta3_elbow_flexion": 0.0}
                reach_error = str(ve)
                success = False
                diagnostics_msgs.append(f"Limits Violation at step {step_idx}: Destination ({coord_target['x']}, {coord_target['y']}, {coord_target['z']}) is out of workspace reach envelop envelope.")

            # Perform physical collision verification checks
            conflicting_bodies = self.world_model.detect_collisions(
                coord_target["x"], coord_target["y"], coord_target["z"],
                ignore_name=source_name if act_type == "grasp" else None
            )

            is_colliding = len(conflicting_bodies) > 0
            if is_colliding:
                success = False
                diagnostics_msgs.append(f"Collision Hazard at step {step_idx} with physical bodies: {conflicting_bodies}")

            # Calculate mechanical active pressure lock states
            vacuum_engaged = 1 if act_type in ["grasp", "move", "release"] and step_idx <= 3 else 0

            # Step metadata
            trajectory_steps_processed.append({
                "step_num": step_idx,
                "action": act_type,
                "description": desc,
                "coordinates": coord_target,
                "joints": joint_angles,
                "vacuum_state": vacuum_engaged,
                "collision_detected": is_colliding,
                "reach_error": reach_error
            })

        # ----------------- STAGE 5: Action Execution & Update State simulation -----------------
        if success:
            # Physically place the item in world coordinates model
            self.world_model.update_object_position(
                source_name, target_obj.x, target_obj.y, target_obj.z + 10.0
            )
            # Commit updated layout to permanent SQLite store
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                UPDATE workspace_objects 
                SET loc_x = ?, loc_y = ?, loc_z = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE item_name = ?
                """, (target_obj.x, target_obj.y, target_obj.z + 10.0, source_name.replace("_", " ")))
                conn.commit()
                logger.info(f"Simulated execution succeeded. {source_name} relocated above {target_name}.")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error executing SQLite position updates: {e}")
            finally:
                conn.close()
        else:
            logger.error(f"Kinematic flight checks invalid. Execution aborted. Failures: {diagnostics_msgs}")

        # ----------------- STAGE 6: Memory storage transactions logging -----------------
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # 1. Log overall task execution status
            cursor.execute("""
            INSERT INTO task_history (task_id, raw_command, interpreted_goal, target_object, destination, execution_status, execution_time_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                raw_command,
                f"Pick up {source_name} and secure within {target_name}",
                source_name,
                target_name,
                "SUCCESS" if success else "FAILED",
                raw_plan["meta_telemetry"]["estimated_duration_sec"]
            ))

            # 2. Log discrete microstep trajectories telemetry measurements
            for t_step in trajectory_steps_processed:
                cursor.execute("""
                INSERT INTO trajectory_telemetry (task_id, sequence_step, joint_1_deg, joint_2_deg, joint_3_deg, vacuum_state)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    task_id,
                    t_step["step_num"],
                    t_step["joints"]["theta1_base_yaw"],
                    t_step["joints"]["theta2_shoulder_pitch"],
                    t_step["joints"]["theta3_elbow_flexion"],
                    t_step["vacuum_state"]
                ))
            conn.commit()
            logger.info(f"Successfully processed persistent storage logs for {task_id}.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Database logging failure during executor commits: {e}")
        finally:
            conn.close()

        summary_msg = "Robot kinematics trajectories calculated and validated successfully. Action sequence executed closed-loop." if success else "Action sequence failed during safety/reach limits verification validation checks."

        return {
            "task_id": task_id,
            "raw_command": raw_command,
            "source": source_name,
            "target": target_name,
            "actions": parsed_cmd.actions,
            "detected_objects": perceived_items,
            "trajectory_steps": trajectory_steps_processed,
            "success": success,
            "message": f"{summary_msg} Details: {'; '.join(diagnostics_msgs) if diagnostics_msgs else 'None'}"
        }


if __name__ == "__main__":
    # Test reasoning engine pipeline standalone
    engine = AutonomousReasoningEngine()
    test_cmd = "Pick up the red cube and place it inside the blue box"
    res = engine.run_pipeline(test_cmd)
    import json
    print(json.dumps(res, indent=2))
