from utils.utilities import calculate_iou

class BoundingBoxAgent:
    def __init__(self, bbox_id, bbox_data, blackboard):
        self.bbox_id = bbox_id
        self.bbox_data = bbox_data
        self.blackboard = blackboard

    def analyze_and_propose(self):
        # Check for overlaps with other boxes
        overlaps = []
        for key, data in self.blackboard.read_all().items():
            if self.check_overlap(self.bbox_data['box'], data['box']) and key != self.bbox_id:
                overlaps.append((key, data))

        # If overlaps exist, decide on fusion
        if overlaps:
            self.fuse_boxes(overlaps)
        else:
            self.blackboard.post(self.bbox_id, self.bbox_data)

    def check_overlap(self, box1, box2, iou_threshold=0.5):
        iou = calculate_iou(box1, box2)
        return iou > iou_threshold

    def fuse_boxes(self, overlapping_boxes):
        # Initialize fused box
        fused_box = {
            'box': [0, 0, 0, 0],
            'score': 0,
            'label': self.bbox_data['label']
        }
        total_score = self.bbox_data['score']
        for _, data in overlapping_boxes:
            total_score += data['score']

        # Calculate weighted averages
        fused_box['box'][0] = sum(data['box'][0] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score
        fused_box['box'][1] = sum(data['box'][1] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score
        fused_box['box'][2] = sum(data['box'][2] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score
        fused_box['box'][3] = sum(data['box'][3] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score

        # Calculate average score
        fused_box['score'] = total_score / (len(overlapping_boxes) + 1)

        # Post the fused box to the blackboard
        self.blackboard.post(f'fused_{self.bbox_id}', fused_box)
