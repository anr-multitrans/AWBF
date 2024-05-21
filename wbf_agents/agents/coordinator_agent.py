class CoordinatorAgent:
    def __init__(self, blackboard):
        self.blackboard = blackboard

    def review_and_decide(self):
        # Collect final fused boxes from the blackboard
        fused_boxes, scores, labels = [], [], []
        for key, data in self.blackboard.read_all().items():
            if key.startswith('fused_'):
                fused_boxes.append(data['box'])
                scores.append(data['score'])
                labels.append(data['label'])

        # Post final fused boxes to the blackboard
        self.blackboard.post("final_fused_boxes", {"boxes": fused_boxes, "scores": scores, "labels": labels})
