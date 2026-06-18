import os
import json
import time
from google import genai
from google.genai import types

class VisionLanguageActionPlanner:
    _cooldown_until = 0.0

    def __init__(self):
        """Initializes the server-side Gemini system client."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def formulate_vla_sequence(self, command: str, detected_objects: list):
        """
        Interrogates the model to transform fuzzy command instructions into
        structured Cartesian steps using real-time coordinates.
        """
        if not self.client or time.time() < self._cooldown_until:
            # High-fidelity Simulation fall-back pattern
            fallback_plan = self._simulate_planning_brain(command, detected_objects)
            fallback_plan["mode"] = "fallback_simulation"
            return fallback_plan

        system_instruction = """
        You are the Vision-Language-Action (VLA) controller of a simulated 3-DOF robot manipulator arm.
        The tabletop workstation boundaries are constrained to: X [-150 to 150] and Y [100 to 220]. 
        Your goal is to parse fuzzy natural language commands, identify target and receptor objects,
        and generate a 7-step sequence list of physical joint travel plans to transport items accurately.
        
        Ensure steps follow this chronological progression:
        1. "locate" - Run camera analysis on coordinates.
        2. "move_to" - Float above the target (Z offset of +50mm).
        3. "grasp" - Lower arm and engage vacuum clamp layout.
        4. "lift" - Elevate target securely (+80mm height clearing threshold).
        5. "transport" - Execute smooth movement towards destination.
        6. "release" - Drop payload gently inside coordinates.
        7. "home" - Revert joint link segments to resting home pose (0, 120, 100).
        """

        prompt = f"""
        Analyze operator command request: "{command}"
        
        Workspace items logs extracted from SQLite persistence:
        {json.dumps(detected_objects, indent=2)}
        
        Map the command logically:
        - Determine "targetObject" matching name.
        - Determine "targetReceptor" matching destination box.
        - Generate the 7 steps array utilizing coordinates from the table records.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "understoodCommand": {"type": "STRING"},
                            "targetObject": {"type": "STRING"},
                            "targetReceptor": {"type": "STRING"},
                            "planSteps": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "stepNum": {"type": "INTEGER"},
                                        "action": {"type": "STRING"},
                                        "description": {"type": "STRING"},
                                        "targetX": {"type": "NUMBER"},
                                        "targetY": {"type": "NUMBER"},
                                        "targetZ": {"type": "NUMBER"}
                                    },
                                    "required": ["stepNum", "action", "description", "targetX", "targetY", "targetZ"]
                                }
                            }
                        },
                        "required": ["understoodCommand", "targetObject", "targetReceptor", "planSteps"]
                    }
                )
            )
            
            plan = json.loads(response.text)
            plan["mode"] = "cognitive_ai"
            return plan

        except Exception as e:
            error_str = str(e)
            if any(term in error_str.lower() for term in ["quota", "exhausted", "429"]):
                # Cooldown for 5 minutes
                VisionLanguageActionPlanner._cooldown_until = time.time() + 300.0
            # Graceful robust log fallback
            fallback_plan = self._simulate_planning_brain(command, detected_objects)
            fallback_plan["mode"] = "fallback_simulation"
            fallback_plan["error"] = error_str
            return fallback_plan

    def _simulate_planning_brain(self, cmd: str, objects: list):
        """Simulates precise planning trajectories locally using keyword matching logic."""
        cleaned = cmd.lower()
        
        # Simple logical keyword parser
        target_name = "red cube"
        receptor_name = "blue box"
        
        # Pick matching physical models
        if "green" in cleaned or "sphere" in cleaned:
            target_name = "green sphere"
        elif "orange" in cleaned or "pyramid" in cleaned:
            target_name = "orange pyramid"
        elif "yellow" in cleaned or "container" in cleaned:
            target_name = "yellow container"
        elif "blue" in cleaned:
            target_name = "blue box"

        if "yellow" in cleaned and ("container" in cleaned or "box" in cleaned):
            receptor_name = "yellow container"
        elif "red" in cleaned:
            receptor_name = "red cube"
        else:
            receptor_name = "blue box"

        # Search coordinates in data list catalog
        target_obj = next((o for o in objects if o["name"] == target_name), objects[0])
        receptor_obj = next((o for o in objects if o["name"] == receptor_name), objects[1])

        return {
            "understoodCommand": f"Formulated sequence to pick up the {target_name} and deposit into {receptor_name}.",
            "targetObject": target_name,
            "targetReceptor": receptor_name,
            "mode": "simulation",
            "planSteps": [
                {
                    "stepNum": 1,
                    "action": "locate",
                    "description": f"Verifying contours centering and distance markers of {target_name}.",
                    "targetX": target_obj["x"],
                    "targetY": target_obj["y"],
                    "targetZ": target_obj["z"]
                },
                {
                    "stepNum": 2,
                    "action": "move_to",
                    "description": f"Interpolating hover coordinates sequence 50mm above {target_name}.",
                    "targetX": target_obj["x"],
                    "targetY": target_obj["y"],
                    "targetZ": target_obj["z"] + 50
                },
                {
                    "stepNum": 3,
                    "action": "grasp",
                    "description": f"Actuating gripper clamp. Validating grip torque constraints on {target_name}.",
                    "targetX": target_obj["x"],
                    "targetY": target_obj["y"],
                    "targetZ": target_obj["z"]
                },
                {
                    "stepNum": 4,
                    "action": "lift",
                    "description": f"Lifting {target_name} vertically (+80mm) to clear adjacent obstacles.",
                    "targetX": target_obj["x"],
                    "targetY": target_obj["y"],
                    "targetZ": target_obj["z"] + 80
                },
                {
                    "stepNum": 5,
                    "action": "transport",
                    "description": f"Path planning active. Carrying cargo seamlessly to hover configuration over {receptor_name}.",
                    "targetX": receptor_obj["x"],
                    "targetY": receptor_obj["y"],
                    "targetZ": receptor_obj["z"] + 60
                },
                {
                    "stepNum": 6,
                    "action": "release",
                    "description": f"Releasing payload pressure locks inside target {receptor_name} container space.",
                    "targetX": receptor_obj["x"],
                    "targetY": receptor_obj["y"],
                    "targetZ": receptor_obj["z"]
                },
                {
                    "stepNum": 7,
                    "action": "home",
                    "description": "Trajectory cleared. Returning robot mechanical joints to resting home pos.",
                    "targetX": 0,
                    "targetY": 120,
                    "targetZ": 100
                }
            ]
        }
