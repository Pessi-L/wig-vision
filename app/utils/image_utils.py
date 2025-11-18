import cv2

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=4):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image