import cv2
import numpy as np

class DataProcessingAgent:
    def preprocess(self, image_data):
        """
        Preprocess the input image data.
        Example: Resize the image and normalize pixel values.

        :param image_data: Input image data (numpy array)
        :return: Preprocessed image data
        """
        # Example: Resize the image to 640x640 and normalize pixel values
        image_data = cv2.resize(image_data, (640, 640))
        image_data = image_data / 255.0  # Normalize pixel values to [0, 1]
        return image_data

    def postprocess(self, fused_boxes, score_threshold=0.5):
        """
        Postprocess the fused bounding boxes.
        Example: Apply a score threshold and format the output.

        :param fused_boxes: Dictionary containing fused boxes, scores, and labels
        :param score_threshold: Confidence score threshold
        :return: Filtered and formatted bounding boxes
        """
        boxes, scores, labels = fused_boxes['boxes'], fused_boxes['scores'], fused_boxes['labels']
        filtered_boxes = []
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                filtered_boxes.append({
                    "box": box,
                    "score": score,
                    "label": label
                })
        
        return filtered_boxes
