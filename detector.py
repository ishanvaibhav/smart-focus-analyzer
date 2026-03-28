"""
detector.py — Face + Eye Attention Detection
=============================================
Uses OpenCV Haar Cascades (built-in, no download needed).

Detects:
  • Is there a face in the frame?          → if not, DISTRACTED
  • Is the face centred (looking at screen)? → if not, DISTRACTED
  • Are eyes open?                          → if closed, DROWSY
"""

import cv2


class FaceAttentionDetector:
    """
    Analyses a single image frame and returns a focus status.

    Usage:
        detector = FaceAttentionDetector()
        result   = detector.analyze(frame)
        print(result["status"])   # "focused" | "distracted" | "drowsy"
    """

    def __init__(self):
        # Haar Cascade XML files come BUNDLED with OpenCV — no download needed
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # ── Tunable settings (can be changed from the Streamlit sidebar) ──
        self.CENTER_TOLERANCE    = 0.20   # Max allowed deviation from screen centre
        self.EYE_CLOSED_THRESHOLD = 15    # Consecutive frames with no eyes = drowsy

        # Internal counter — how many frames in a row have eyes been missing?
        self._closed_eye_count = 0

    # ──────────────────────────────────────────────────────────────────────
    def analyze(self, frame, draw=True, show_eyes=True):
        """
        Analyse one frame.

        Args:
            frame     : BGR numpy array (from cv2.VideoCapture or PIL conversion)
            draw      : draw bounding boxes on the frame
            show_eyes : also draw eye boxes

        Returns:
            dict with keys:
              "status"  → "focused" | "distracted" | "drowsy"
              "reason"  → human-readable explanation
              "frame"   → annotated frame (same object, modified in-place)
        """

        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_width  = frame.shape[1]
        frame_centre = frame_width / 2

        # ── STEP 1: Detect faces ────────────────────────────────────────
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # Scale step between scans (1.1 = 10% smaller each step)
            minNeighbors=5,      # How many neighbours a detection needs to be kept
            minSize=(60, 60)     # Ignore faces smaller than 60×60 pixels
        )

        # No face → user looked away or left frame
        if len(faces) == 0:
            self._closed_eye_count = 0
            if draw:
                self._put_label(frame, "No face detected", (15, frame.shape[0] - 15),
                                color=(0, 60, 255))
            return {
                "status": "distracted",
                "reason": "No face detected in frame",
                "frame" : frame
            }

        # ── STEP 2: Pick the largest face (most likely the user) ────────
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_centre = x + w / 2

        if draw:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            self._put_label(frame, "Face", (x, y - 8), color=(0, 220, 0))

        # ── STEP 3: Is the face centred on screen? ──────────────────────
        deviation = abs(face_centre - frame_centre) / frame_width
        if deviation > self.CENTER_TOLERANCE:
            self._closed_eye_count = 0
            return {
                "status": "distracted",
                "reason": f"Looking away from screen ({deviation:.0%} off-centre)",
                "frame" : frame
            }

        # ── STEP 4: Detect eyes inside the face region ──────────────────
        # Crop to just the face (much faster + more accurate than full frame)
        face_gray  = gray[y: y + h, x: x + w]
        face_color = frame[y: y + h, x: x + w]   # This is a VIEW of frame (not a copy)

        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20)
        )

        # Keep only eyes in the TOP HALF of the face
        # (avoids false detections on nose/mouth/beard)
        eyes = [e for e in eyes if e[1] < h // 2]

        if show_eyes and draw:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh),
                              (255, 200, 0), 1)

        # ── STEP 5: Drowsiness check ────────────────────────────────────
        # If eyes haven't been visible for several frames → drowsy
        if len(eyes) < 2:
            self._closed_eye_count += 1
        else:
            # Slowly reset the counter when eyes become visible again
            self._closed_eye_count = max(0, self._closed_eye_count - 2)

        if self._closed_eye_count >= self.EYE_CLOSED_THRESHOLD:
            return {
                "status": "drowsy",
                "reason": f"Eyes closed for {self._closed_eye_count} frames — take a break!",
                "frame" : frame
            }

        # ── STEP 6: All checks passed → focused ─────────────────────────
        eyes_str = f"{len(eyes)} eye{'s' if len(eyes) != 1 else ''} detected"
        return {
            "status": "focused",
            "reason": f"Face centred, {eyes_str}",
            "frame" : frame
        }

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _put_label(frame, text, pos, color=(255, 255, 255)):
        """Helper: draws text with a dark shadow so it's readable on any background."""
        shadow_pos = (pos[0] + 1, pos[1] + 1)
        cv2.putText(frame, text, shadow_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
