import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


class ObjectDetector:
    def __init__(self, pixel_to_mm_ratio: float = 1.25, frame_width: int = 640, frame_height: int = 480):
        """
        Initializes the HSV Calibration thresholds and camera transformation factors.
        Standard calibration center is (320, 240).
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.cx_calib = frame_width // 2    # 320 px
        self.cy_calib = frame_height // 2   # 240 px

        # Define rigid HSV ranges for tabletop blocks
        self.color_ranges = {
            "red_cube": {
                # Red has wrapping thresholds on the HSV spectrum
                "lower1": np.array([0, 120, 70]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([170, 120, 70]),
                "upper2": np.array([180, 255, 255]),
                "display_name": "Red Cube"
            },
            "blue_box": {
                "lower1": np.array([100, 150, 50]),
                "upper1": np.array([140, 255, 255]),
                "display_name": "Blue Box"
            },
            "green_sphere": {
                "lower1": np.array([35, 100, 100]),
                "upper1": np.array([85, 255, 255]),
                "display_name": "Green Sphere"
            }
        }

    def pixel_to_mm(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """
        Converts 2D camera viewport pixel locations to millimeters relative to workbase base center.
        Equations matching:
          - X increases to the right (+X)
          - Y increases going upwards (-Y in image coords, so we subtract from calibration center)
        """
        mm_x = (px_x - self.cx_calib) * self.pixel_to_mm_ratio
        mm_y = (self.cy_calib - px_y) * self.pixel_to_mm_ratio
        return round(mm_x, 2), round(mm_y, 2)

    def detect_all_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs OpenCV contours pipeline across HSV color spectrums.
        Returns a list of detected objects in the format requested.
        """
        if frame is None:
            return []

        detected_list = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply smoothing to eliminate pixel flare
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        for obj_key, threshold in self.color_ranges.items():
            # Apply color masks
            if "lower2" in threshold:
                # Wrap-around red masking combination
                mask1 = cv2.inRange(blurred, threshold["lower1"], threshold["upper1"])
                mask2 = cv2.inRange(blurred, threshold["lower2"], threshold["upper2"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(blurred, threshold["lower1"], threshold["upper1"])

            # Morphological opening & closing to denoise tabletop specular reflections
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find object contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 150: # Ignore noise specs
                    continue

                # Math moment of contour to establish centroid pixel coordinate
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                # Translate coordinates to mm space
                mm_x, mm_y = self.pixel_to_mm(cx, cy)

                # Classify shape characteristics to estimate confidence
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
                
                # Aspect Ratio analysis
                rect = cv2.minAreaRect(contour)
                (x, y), (w, h), angle = rect
                aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0

                # Calculate confidence index based on color-density consistency and target-shape expectations
                confidence = 0.95
                if obj_key == "green_sphere":
                    # Sphere expects high circularity (ideally close to 1.0)
                    shape_score = max(0.0, 1.0 - abs(1.0 - circularity))
                    confidence = 0.7 + (0.28 * shape_score)
                elif obj_key == "red_cube":
                    # Cube expects square aspect ratio and moderate solidity
                    shape_score = aspect_ratio
                    confidence = 0.8 + (0.18 * shape_score)
                elif obj_key == "blue_box":
                    # Box allows slightly rectangular forms
                    confidence = 0.85 + (0.12 * aspect_ratio)

                confidence = round(float(min(1.0, max(0.0, confidence))), 2)

                detected_list.append({
                    "object_name": threshold["display_name"],
                    "x": mm_x,
                    "y": mm_y,
                    "confidence": confidence
                })

        return detected_list



class TabletopCentroidTracker:
    def __init__(self):
        """Wrapper constructor matching old camera systems."""
        from vision.camera import TabletopCamera
        self.camera = TabletopCamera()
        self.detector = ObjectDetector()

    def process_mock_frame(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Captures camera coordinates and returns them in the old legacy dictionary format."""
        frame, _ = self.camera.read_frame()
        raw_items = self.detector.detect_all_objects(frame)
        
        legacy_list = []
        pixel_to_mm = 1.25
        for item in raw_items:
            display_name = item["object_name"]
            normalized_name = display_name.lower()
            mm_x = item["x"]
            mm_y = item["y"]
            
            # Compute estimated reverse camera pixels (u, v) based on origin 320, 240
            cx = int(320 + (mm_x / pixel_to_mm))
            cy = int(240 - (mm_y / pixel_to_mm))
            
            legacy_list.append({
                "name": normalized_name,
                "cx": cx,
                "cy": cy,
                "x": mm_x,
                "y": mm_y,
                "z": 10.0 if "sphere" in normalized_name or "cube" in normalized_name else 20.0
            })
        return frame, legacy_list


if __name__ == "__main__":
    # Self-test loop
    detector = ObjectDetector()
    # Create simple synthetic test image (black frame with a red square)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (340, 240), (0, 0, 255), -1) # Red Cube mockup in BGR space
    results = detector.detect_all_objects(img)
    print("Self-Test Detection Results:")
    print(results)

