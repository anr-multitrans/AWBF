def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is represented by a list of coordinates [x1, y1, x2, y2].

    :param box1: List of coordinates [x1, y1, x2, y2]
    :param box2: List of coordinates [x1, y1, x2, y2]
    :return: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou