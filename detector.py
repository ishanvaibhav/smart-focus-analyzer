"""
detector.py — Face + Eye Attention Detection
=============================================
Uses OpenCV Haar Cascades (built-in, no download needed).
"""

import cv2


class FaceAttentionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.CENTER_TOLERANCE     = 0.20
        self.EYE_CLOSED_THRESHOLD = 15
        self._closed_eye_count    = 0

    def analyze(self, frame, draw=True, show_eyes=True):
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_width  = frame.shape[1]
        frame_centre = frame_width / 2

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            self._closed_eye_count = 0
            return {"status": "distracted",
                    "reason": "No face detected in frame", "frame": frame}

        x, y, w, h  = max(faces, key=lambda f: f[2] * f[3])
        face_centre  = x + w / 2

        if draw:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            self._put_label(frame, "Face", (x, y - 8), (0, 220, 0))

        deviation = abs(face_centre - frame_centre) / frame_width
        if deviation > self.CENTER_TOLERANCE:
            self._closed_eye_count = 0
            return {"status": "distracted",
                    "reason": f"Looking away ({deviation:.0%} off-centre)",
                    "frame": frame}

        face_gray  = gray[y: y + h, x: x + w]
        face_color = frame[y: y + h, x: x + w]

        eyes = self.eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
        )
        eyes = [e for e in eyes if e[1] < h // 2]

        if show_eyes and draw:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey),
                              (ex + ew, ey + eh), (255, 200, 0), 1)

        if len(eyes) < 2:
            self._closed_eye_count += 1
        else:
            self._closed_eye_count = max(0, self._closed_eye_count - 2)

        if self._closed_eye_count >= self.EYE_CLOSED_THRESHOLD:
            return {"status": "drowsy",
                    "reason": f"Eyes closed for {self._closed_eye_count} frames",
                    "frame": frame}

        return {"status": "focused",
                "reason": f"Face centred, {len(eyes)} eye(s) detected",
                "frame": frame}

    @staticmethod
    def _put_label(frame, text, pos, color=(255, 255, 255)):
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
