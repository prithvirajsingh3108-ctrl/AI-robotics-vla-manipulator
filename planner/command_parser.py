import re
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class ParsedCommand(BaseModel):
    source: str = Field(..., description="The object identifier to manipulate, e.g. 'red cube'")
    destination: str = Field(..., description="The destination container or coordinate anchor, e.g. 'blue box'")
    target: str = Field("", description="Compatibility layer for old target field key")
    actions: List[str] = Field(
        default_factory=lambda: ["locate", "move", "grasp", "move", "release"],
        description="Chronological step actions list required to complete the operation"
    )

    def __init__(self, **data):
        # Synchronize destination and target fields seamlessly
        if "destination" in data and ("target" not in data or not data["target"]):
            data["target"] = data["destination"].replace(" ", "_")
        if "target" in data and ("destination" not in data or not data["destination"]):
            data["destination"] = data["target"].replace("_", " ")
        super().__init__(**data)


class CommandParser:
    def __init__(self, world_model: Optional[Any] = None):
        """
        Initializes the Natural Language Command Parser.
        Tracks known color-space definitions and geometric object catalogs to map terms.
        """
        self.colors = ["red", "blue", "green", "yellow", "orange"]
        self.shapes = ["cube", "box", "sphere", "container", "pyramid"]
        
        # Action mappings corresponding to specific requests if the operator overrides actions
        self.default_actions = ["locate", "move", "grasp", "move", "release"]
        self.world_model = world_model

    def _extract_object_tokens(self, text: str, world_model: Optional[Any] = None) -> List[str]:
        """
        Extracts source and destination object tokens from raw input based on their chronological order of appearance.
        Uses active world model names if available; falls back to color-shape matching patterns.
        """
        normalized = text.lower().replace("_", " ")
        found_matches = []
        active_world = world_model or self.world_model

        # 1. Check world model registered objects first to support dynamic names
        if active_world:
            try:
                # Extract actual objects names in lower case with spaces instead of underscores
                registered_names = [obj.name.replace("_", " ").lower() for obj in active_world.retrieve_active_objects()]
                # Sort names by descending length so complex/long names match before subsets (e.g. "yellow container" before "container")
                registered_names.sort(key=len, reverse=True)
                
                index_matches = []
                for name in registered_names:
                    # Find and keep track of actual indices of occurrences
                    escaped_name = re.escape(name)
                    for match in re.finditer(rf"\b{escaped_name}\b", normalized):
                        start_idx = match.start()
                        # Avoid matching overlapping patterns
                        if not any(abs(start_idx - existing_idx) < len(name) for existing_idx, _ in index_matches):
                            index_matches.append((start_idx, name))
                
                # Sort indices of matches to retain original chronological sequence
                index_matches.sort()
                found_matches = [name for _, name in index_matches]
            except Exception:
                found_matches = []

        # 2. Pattern matchmaking fallback: if we have less than 2 matches, overlay standard color+shape combos
        if len(found_matches) < 2:
            color_shape_matches = []
            pattern = r"\b(red|blue|green|yellow|orange)\s+(cube|box|sphere|container|pyramid)\b"
            for match in re.finditer(pattern, normalized):
                color = match.group(1)
                shape = match.group(2)
                start_idx = match.start()
                token = f"{color} {shape}"
                # Deduplicate if dynamic matching already caught this segment
                if not any(token in fm or fm in token for fm in found_matches):
                    color_shape_matches.append((start_idx, token))
            
            # Combine current known lists retaining textual position order
            combined = []
            for fm in found_matches:
                idx = normalized.find(fm)
                if idx != -1:
                    combined.append((idx, fm))
            for idx, token in color_shape_matches:
                if not any(token in item[1] or item[1] in token for item in combined):
                    combined.append((idx, token))
                    
            combined.sort()
            found_matches = [item[1] for item in combined]

        # 3. Singleton color mapping fallback
        if len(found_matches) < 2:
            single_matches = []
            for color in self.colors:
                pattern = rf"\b{color}\b"
                for match in re.finditer(pattern, normalized):
                    start_idx = match.start()
                    default_map = {
                        "red": "red cube",
                        "blue": "blue box",
                        "green": "green sphere",
                        "yellow": "yellow container",
                        "orange": "orange pyramid"
                    }
                    token = default_map.get(color, f"{color} cube")
                    single_matches.append((start_idx, token))
            
            combined = []
            for fm in found_matches:
                idx = normalized.find(fm)
                if idx != -1:
                    combined.append((idx, fm))
            for idx, token in single_matches:
                if not any(token in item[1] or item[1] in token for item in combined):
                    combined.append((idx, token))
                    
            combined.sort()
            found_matches = [item[1] for item in combined]

        return found_matches

    def parse_instruction(self, command: str, world_model: Optional[Any] = None) -> ParsedCommand:
        """
        Transforms loose operator language into machine-actionable semantic mappings.
        Identifies source item to pick up and target container destination anchor.
        """
        cleaned = command.lower().strip()
        active_world = world_model or self.world_model
        tokens = self._extract_object_tokens(cleaned, active_world)
        
        # Determine defaults if elements are missing
        source_obj = "red cube"
        target_obj = "blue box"
        
        if len(tokens) >= 2:
            source_obj = tokens[0]
            target_obj = tokens[1]
        elif len(tokens) == 1:
            source_obj = tokens[0]
            if source_obj == "blue box":
                source_obj = "red cube"
                target_obj = "blue box"
            else:
                target_obj = "blue box"

        # Check for dynamic overrides and system presets
        actions = list(self.default_actions)
        if "scan" in cleaned or "calibrate" in cleaned:
            actions = ["locate"] + actions
        if "home" in cleaned or "reset" in cleaned:
            actions.append("home")

        return ParsedCommand(
            source=source_obj,
            destination=target_obj,
            actions=actions
        )


if __name__ == "__main__":
    # Standard engineering validation test
    parser = CommandParser()
    sample = "Pick up the red cube and place it inside the blue box"
    result = parser.parse_instruction(sample)
    print("Test Parsing Output:")
    print(result.model_dump_json(indent=2))
