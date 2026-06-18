import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from planner.command_parser import CommandParser, ParsedCommand
from planner.world_model import WorldModel, WorkspaceObject

# Configure system diagnostics logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TaskPlanner")


class PlannedStepTrajectory(BaseModel):
    step_index: int = Field(..., description="Chronological sequence number of physical execution")
    action_type: str = Field(..., description="Action category (locate, move, grasp, release, home)")
    description: str = Field(..., description="Human-readable dynamic log descriptor")
    target_coordinates: dict = Field(..., description="Cartesian millimeter coordinate state {x, y, z}")


class CompletePlanOutput(BaseModel):
    source: str = Field(..., description="The object identifier to manipulate, e.g. 'red_cube'")
    target: str = Field(..., description="The destination container or coordinate anchor, e.g. 'blue_box'")
    actions: List[str] = Field(..., description="List of logical actions resolved")
    estimated_duration_sec: float = Field(0.0, description="Computed mechanical travel overhead calculation")
    geometric_steps: List[PlannedStepTrajectory] = Field(
        default_factory=list,
        description="Resolved Cartesian 3D trajectories for robot actuator joints"
    )


class TaskPlanner:
    def __init__(self, parser: Optional[CommandParser] = None, world: Optional[WorldModel] = None):
        """
        Orchestration engine coordinating natural language comprehension (CommandParser) 
        and spatial desktop geometry constraints (WorldModel) to assemble path trajectories.
        """
        self.parser = parser or CommandParser()
        self.world = world or WorldModel()

    def generate_task_plan(self, command_text: str) -> Dict[str, Any]:
        """
        Processes operator instructions, integrates spatial lookups, and returns
        the high-fidelity schema conforming exactly to runtime requirements.
        """
        logger.info(f"Received high-level natural language request: '{command_text}'")

        # 1. Parse command terms
        parsed: ParsedCommand = self.parser.parse_instruction(command_text, self.world)
        
        # 2. Localize objects from the World Model database
        source_obj: Optional[WorkspaceObject] = self.world.get_object(parsed.source)
        target_obj: Optional[WorkspaceObject] = self.world.get_object(parsed.target)

        # Log missing models fallback warnings
        if not source_obj:
            logger.warning(f"Manipulate target '{parsed.source}' missing from real-time database. Seeding simulation model coordinates.")
            # Seeding safe default values inside model
            self.world.register_object(WorkspaceObject(
                name=parsed.source, shape="cube", color_hex="#ef4444", x=80.0, y=150.0, z=10.0
            ))
            source_obj = self.world.get_object(parsed.source)

        if not target_obj:
            logger.warning(f"Placement receptor '{parsed.target}' missing. Registering default target boundary.")
            self.world.register_object(WorkspaceObject(
                name=parsed.target, shape="box", color_hex="#3b82f6", x=-100.0, y=180.0, z=20.0
            ))
            target_obj = self.world.get_object(parsed.target)

        # 3. Formulate detailed kinematic Cartesian path trajectory steps
        geometric_steps = []
        travel_distance_sum = 0.0
        
        # Default start posture (Home base)
        current_x, current_y, current_z = 0.0, 120.0, 100.0

        for idx, action in enumerate(parsed.actions):
            step_num = idx + 1
            desc = ""
            tx, ty, tz = current_x, current_y, current_z

            if action == "locate":
                desc = f"Interrogating computer vision matrix boundaries at target centroid space of {source_obj.name}."
                tx, ty, tz = source_obj.x, source_obj.y, source_obj.z
            elif action == "move":
                # Determine context: first 'move' usually hovers above source, second 'move' hovers above target
                if idx < 3: # Move to source object
                    desc = f"Interpolating hover coordinates 50mm above {source_obj.name} targeting pickup position."
                    tx, ty, tz = source_obj.x, source_obj.y, source_obj.z + 50.0
                else: # Transport move to target
                    desc = f"Carrying secure assembly vector seamlessly to hover range over {target_obj.name} container."
                    tx, ty, tz = target_obj.x, target_obj.y, target_obj.z + 50.0
            elif action == "grasp":
                desc = f"Lowering joint limbs. Actuating vacuum gripper pressure system onto {source_obj.name}."
                tx, ty, tz = source_obj.x, source_obj.y, source_obj.z
            elif action == "release":
                desc = f"Releasing pressure locks inside {target_obj.name} payload space. Disengaging feedback sensory checks."
                tx, ty, tz = target_obj.x, target_obj.y, target_obj.z

            # Accumulate physical travel distance in mm to approximate mechanical overheads
            step_dist = ((tx - current_x) ** 2 + (ty - current_y) ** 2 + (tz - current_z) ** 2) ** 0.5
            travel_distance_sum += step_dist
            
            # Progress trajectory positions
            current_x, current_y, current_z = tx, ty, tz

            geometric_steps.append(
                PlannedStepTrajectory(
                    step_index=step_num,
                    action_type=action,
                    description=desc,
                    target_coordinates={"x": round(tx, 2), "y": round(ty, 2), "z": round(tz, 2)}
                )
            )

        # Calculate estimated duration (assuming average effector travel rate of 50mm/s + 1.2s action delays)
        estimated_duration = round((travel_distance_sum / 50.0) + (len(parsed.actions) * 1.2), 2)

        # Assemble the clean requested high-level schema return packet
        complete_plan = CompletePlanOutput(
            source=parsed.source,
            target=parsed.target,
            actions=parsed.actions,
            estimated_duration_sec=estimated_duration,
            geometric_steps=geometric_steps
        )

        logger.info(f"Formulated trajectory plan containing {len(geometric_steps)} steps. Estimated completion: {estimated_duration}s")
        
        # We output a dictionary matching the rigid API structure specifications
        return {
            "source": complete_plan.source,
            "target": complete_plan.target,
            "actions": complete_plan.actions,
            "meta_telemetry": {
                "estimated_duration_sec": complete_plan.estimated_duration_sec,
                "step_detail_logs": [step.model_dump() for step in complete_plan.geometric_steps]
            }
        }


if __name__ == "__main__":
    # Validate orchestration logic
    planner = TaskPlanner()
    out = planner.generate_task_plan("Pick up the red cube and place it inside the blue box")
    import json
    print(json.dumps(out, indent=2))
