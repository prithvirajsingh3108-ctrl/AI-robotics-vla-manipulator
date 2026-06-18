import cv2
import numpy as np
import random
import time
from typing import Tuple, Optional


class TabletopCamera:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Manages the physical hardware video capture stream.
        Includes high-performance synthetic mockup fallback when running on headless cloud environments.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_mock = False
        
        self.initialize_camera()

    def initialize_camera(self) -> None:
        """Attempts connection to local OpenCV video devices."""
        try:
            # cv2.VideoCapture(index) will try to open webcam index
            self.cap = cv2.VideoCapture(self.camera_index)
            # Set buffer bounds
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check connection
            if not self.cap or not self.cap.isOpened():
                raise ConnectionError("No video capture devices connected.")
            self.is_mock = False
        except Exception:
            # Gracefully activate mock simulation container
            self.is_mock = True
            self.cap = None

    def read_frame(self) -> Tuple[np.ndarray, bool]:
        """
        Grabs active viewport image matrix.
        Returns:
            - frame: BGR raw numpy matrix array
            - is_simulated: boolean indicating if the image was artificially generated.
        """
        if not self.is_mock and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Ensure correct resize
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return frame, False
            else:
                # Device timed out or disconnected, fallback to mock generator
                self.is_mock = True

        # Render simulated calibration frame
        return self._generate_simulated_workbench_frame(), True

    def _generate_simulated_workbench_frame(self) -> np.ndarray:
        """
        Synthesizes a realistic dark workbench stage with visual markers.
        Features minor physical vibration jitter simulating a running mechanical arm.
        """
        # Create dark metallic grey coordinate canvas
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 22
        
        # Draw high-contrast coordinate workspace crosshairs
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (45, 45, 45), 1)
        cv2.line(frame, (0, self.height // 2), (self.width, self.height // 2), (45, 45, 45), 1)

        # Draw focus-area concentric verification circles
        cv2.circle(frame, (self.width // 2, self.height // 2), 150, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.circle(frame, (self.width // 2, self.height // 2), 220, (30, 30, 30), 1, cv2.LINE_AA)
        
        # Minor continuous physical vibration loop values (millimeter jitter)
        jitter_x = random.uniform(-0.6, 0.6)
        jitter_y = random.uniform(-0.6, 0.6)
        pixel_to_mm = 1.25

        # 1. Overlay Red Cube tracking (Position close to: X=80.0, Y=150.0 relative to calibration center 320, 240)
        # Translation equations mapping mm to pixels:
        # px_x = 320 + (mm_x / pixel_to_mm)
        # px_y = 240 - (mm_y / pixel_to_mm)
        red_cx = int(320 + ((80.0 + jitter_x) / pixel_to_mm))
        red_cy = int(240 - ((150.0 + jitter_y) / pixel_to_mm))
        cv2.rectangle(frame, (red_cx - 15, red_cy - 15), (red_cx + 15, red_cy + 15), (0, 0, 240), -1) # BGR red color cube
        cv2.rectangle(frame, (red_cx - 15, red_cy - 15), (red_cx + 15, red_cy + 15), (255, 255, 255), 1) # core white boundary border

        # 2. Overlay Blue Box tracking (Position close to: X=-100.0, Y=180.0)
        blue_cx = int(320 + ((-100.0 + jitter_x) / pixel_to_mm))
        blue_cy = int(240 - ((180.0 + jitter_y) / pixel_to_mm))
        # Draw rectangular cargo enclosure
        cv2.rectangle(frame, (blue_cx - 24, blue_cy - 18), (blue_cx + 24, blue_cy + 18), (250, 0, 0), -1) # box body BGR blue
        cv2.rectangle(frame, (blue_cx - 24, blue_cy - 18), (blue_cx + 24, blue_cy + 18), (100, 255, 100), 1) # calibration green lock

        # 3. Overlay Green Sphere tracking (Position close to: X=120.0, Y=120.0)
        green_cx = int(320 + ((120.0 + jitter_x) / pixel_to_mm))
        green_cy = int(240 - ((120.0 + jitter_y) / pixel_to_mm))
        cv2.circle(frame, (green_cx, green_cy), 14, (0, 220, 0), -1) # green core circle
        cv2.circle(frame, (green_cx, green_cy), 14, (255, 255, 255), 1) # outline

        # Top diagnostic annotations HUD
        cv2.putText(frame, "LIVE CAMERA GRID FEED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, f"STATUS: SIMULATED WEBCAM PIPELINE - RESOLUTION: {self.width}x{self.height}", 
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FOCAL ORIGIN: ({self.width//2}, {self.height//2}) pixels", 
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
        
        return frame

    def release(self) -> None:
        """Kills stream channel clean on terminal exits."""
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    # Smoke validation test
    camera = TabletopCamera()
    frame, is_sim = camera.read_frame()
    print(f"Captured frame dimensions: {frame.shape}, Simulated: {is_sim}")
    camera.release()
