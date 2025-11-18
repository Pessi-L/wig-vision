from app.core.preprocess.face_detectors import MediaPipeFaceDetector

detector = MediaPipeFaceDetector()

from app.utils import image_utils
import cv2


# Test face detection
img_path = '/'
image = cv2.imread(img_path)

face = detector.detect_main_face(image)

result_img = image_utils.draw_boxes(image, face)

cv2.imshow("Face Detection", result_img)
cv2.waitKey(0)



