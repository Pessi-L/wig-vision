from abc import ABC, abstractmethod

class BaseFaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, image, mode='image'):
        """
        :param image:
        :param mode: optional modes: image, video, live_stream
        :return:
        """
        pass

    def detect_main_face(self, image):
        """
        Detects the main face in an image

        Logic:
        - Detect faces
        - Pick the largest bounding box by area.
        - If multiple boxes are similar in size, choose the one closest to the image center.

        :param image: numpy array
        :return: box of the main face
        """
        # Detect faces
        faces = self.detect_faces(image)

        h, w, _ = image.shape
        cx, cy = w/2, h/2

        # Sort by are descending
        faces_sorted = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        largest_area = faces_sorted[0][2] * faces_sorted[0][3]

        # Filter faces with area within 90% of largest
        candidates = [box for box in faces_sorted if box[2] * box[3] >= 0.9 * largest_area]

        # Pick the candidate closest to the center
        def distance_to_center(box):
            bx, by, bw, bh = box
            return (((bx + bw/2) - cx)**2 + ((by + bh/2) - cy)**2)**0.5

        main_face = min(candidates, key=distance_to_center)
        return [main_face]


