from abc import ABC, abstractmethod

class BaseFaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, image):
        """
        :param image: numpy array
        :return:
        """
        pass

