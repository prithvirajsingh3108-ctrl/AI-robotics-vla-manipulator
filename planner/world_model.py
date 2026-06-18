import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WorkspaceObject(BaseModel):
    name: str = Field(..., description="Unique semantic identifier of the physical object, e.g. 'red_cube'")
    shape: str = Field(..., description="Geometric classification category (cube, sphere, box, pyramid, etc.)")
    color_hex: str = Field(..., description="Hex representation of the identified color envelope")
    x: float = Field(..., description="X coordinate in physical workspace space (in millimeters)")
    y: float = Field(..., description="Y coordinate in physical workspace space (in millimeters)")
    z: float = Field(..., description="Z heighten height reference (in millimeters)")
    radius_or_extent: float = Field(15.0, description="Spatial occupancy boundary threshold for collision calculation")


class WorldModel:
    def __init__(self, workspace_limit_x: tuple = (-150.0, 150.0), workspace_limit_y: tuple = (100.0, 220.0)):
        """
        Initializes the dynamic WorldModel containing the live mechanical and tabletop geometry boundaries.
        """
        self.workspace_limit_x = workspace_limit_x
        self.workspace_limit_y = workspace_limit_y
        self.objects: Dict[str, WorkspaceObject] = {}
        self._load_default_blocks()

    def _load_default_blocks(self) -> None:
        """Seeds the workspace model tracking layout references with standard capstone items."""
        defaults = [
            WorkspaceObject(name="red_cube", shape="cube", color_hex="#ef4444", x=80.0, y=150.0, z=10.0, radius_or_extent=15.0),
            WorkspaceObject(name="blue_box", shape="box", color_hex="#3b82f6", x=-100.0, y=180.0, z=20.0, radius_or_extent=25.0),
            WorkspaceObject(name="green_sphere", shape="sphere", color_hex="#22c55e", x=120.0, y=120.0, z=10.0, radius_or_extent=12.0),
            WorkspaceObject(name="yellow_container", shape="box", color_hex="#eab308", x=-60.0, y=140.0, z=20.0, radius_or_extent=25.0),
            WorkspaceObject(name="orange_pyramid", shape="pyramid", color_hex="#f97316", x=40.0, y=190.0, z=15.0, radius_or_extent=15.0),
        ]
        for obj in defaults:
            self.register_object(obj)

    def register_object(self, obj: WorkspaceObject) -> None:
        """Inserts or updates an active tabletop item inside the live registry maps."""
        # Sanitize internal name formatting (standard space/underscore mapping)
        sanitized_name = obj.name.replace(" ", "_").lower()
        obj.name = sanitized_name
        self.objects[sanitized_name] = obj

    def get_object(self, name: str) -> Optional[WorkspaceObject]:
        """Queries the live model to locate target details with name sanitization fallback."""
        sanitized_name = name.replace(" ", "_").lower()
        return self.objects.get(sanitized_name)

    def retrieve_active_objects(self) -> List[WorkspaceObject]:
        """Provides lists of all available physical items inside the active visual frame."""
        return list(self.objects.values())

    def update_object_position(self, name: str, x: float, y: float, z: float) -> bool:
        """
        Relocates item target coordinates with safety clamp metrics conforming to workspace limit bounds.
        """
        obj = self.get_object(name)
        if not obj:
            return False

        # Apply kinematic workspace boundary constraint protections
        min_x, max_x = self.workspace_limit_x
        min_y, max_y = self.workspace_limit_y

        obj.x = max(min_x, min(max_x, x))
        obj.y = max(min_y, min(max_y, y))
        obj.z = max(0.0, z)  # Z boundary protects table surface collision thresholds
        return True

    def calculate_distance(self, name_a: str, name_b: str) -> Optional[float]:
        """Determines physical distance in millimeters between two active tabletop objects."""
        obj_a = self.get_object(name_a)
        obj_b = self.get_object(name_b)
        if not obj_a or not obj_b:
            return None
        return math.sqrt((obj_a.x - obj_b.x) ** 2 + (obj_a.y - obj_b.y) ** 2 + (obj_a.z - obj_b.z) ** 2)

    def detect_collisions(self, test_x: float, test_y: float, test_z: float, ignore_name: Optional[str] = None) -> List[str]:
        """
        Evaluates potential kinematic collisions against physical coordinate spaces of occupied shapes.
        Returns names of conflicting objects.
        """
        collisions = []
        for name, obj in self.objects.items():
            if ignore_name and name == ignore_name.replace(" ", "_").lower():
                continue
            
            # Simple bounding cylinder distance check
            plane_dist = math.sqrt((obj.x - test_x) ** 2 + (obj.y - test_y) ** 2)
            height_diff = abs(obj.z - test_z)
            
            # If coordinates are overlapping physical extent boundaries
            if plane_dist < obj.radius_or_extent and height_diff < 30.0:
                collisions.append(name)
        return collisions
