import mediapipe as mp
import cv2
from base_detector import BaseFaceDetector

class MediaPipeFaceDetector(BaseFaceDetector):

    def __init__(self, min_detection_confidence=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        faces = []
        if results.detections:
            h, w, _ = image.shape
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, by = int(box.width * w), int(box.height * h)
                faces.append([x, y, bw, by])
        return faces


    def detect_main_face(self, image):
        faces = self.detect_faces(image)
        return super().detect_main_face(faces)