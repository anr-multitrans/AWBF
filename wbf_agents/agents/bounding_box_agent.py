from utils.utilities import calculate_iou

class BoundingBoxAgent:
    def __init__(self, bbox_id, bbox_data, blackboard):
        self.bbox_id = bbox_id
        self.bbox_data = bbox_data
        self.blackboard = blackboard

    def analyze_and_propose(self):
        print(f"BoundingBoxAgent {self.bbox_id}: Analyzing bounding box {self.bbox_data}")
        overlaps = []
        for key, data in self.blackboard.read_all().items():
            if key != self.bbox_id and self.check_overlap(self.bbox_data['box'], data['box']):
                overlaps.append((key, data))
                print(f"BoundingBoxAgent {self.bbox_id}: Found overlap with {key}")

        if overlaps:
            self.fuse_boxes(overlaps)
        else:
            print(f"BoundingBoxAgent {self.bbox_id}: No overlaps found, posting to blackboard")
            self.blackboard.post(self.bbox_id, self.bbox_data)

    def check_overlap(self, box1, box2, iou_threshold=0.5):
        iou = calculate_iou(box1, box2)
        print(f"Checking overlap between {box1} and {box2}, IoU: {iou}")
        return iou > iou_threshold

    def fuse_boxes(self, overlapping_boxes):
        print(f"BoundingBoxAgent {self.bbox_id}: Fusing boxes {overlapping_boxes}")
        fused_box = {
            'box': [0, 0, 0, 0],
            'score': 0,
            'label': self.bbox_data['label']
        }
        total_score = self.bbox_data['score']
        for _, data in overlapping_boxes:
            total_score += data['score']

        fused_box['box'][0] = sum(data['box'][0] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score
        fused_box['box'][1] = sum(data['box'][1] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score
        fused_box['box'][2] = sum(data['box'][2] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score
        fused_box['box'][3] = sum(data['box'][3] * data['score'] for _, data in overlapping_boxes + [(self.bbox_id, self.bbox_data)]) / total_score

        fused_box['score'] = total_score / (len(overlapping_boxes) + 1)

        print(f"BoundingBoxAgent {self.bbox_id}: Posting fused box {fused_box} to blackboard")
        self.blackboard.post(f'fused_{self.bbox_id}', fused_box)
