from ensemble_boxes import weighted_boxes_fusion

class CoordinatorAgent:
    def __init__(self, blackboard):
        self.blackboard = blackboard

    def review_and_decide(self):
        # Gather all box data
        boxes, scores, labels = [], [], []
        for key, data in self.blackboard.data.items():
            boxes.append(data['box'])
            scores.append(data['score'])
            labels.append(data['label'])

        # Fusion logic using WBF
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
        self.blackboard.post("fused_boxes", {"boxes": boxes, "scores": scores, "labels": labels})
