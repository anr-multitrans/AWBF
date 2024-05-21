class CoordinatorAgent:
    def __init__(self, blackboard):
        self.blackboard = blackboard

    def review_and_decide(self):
        print("CoordinatorAgent: Reviewing fused boxes from blackboard")
        fused_boxes, scores, labels = [], [], []
        for key, data in self.blackboard.read_all().items():
            if isinstance(key, str) and key.startswith('fused_'):
                fused_boxes.append(data['box'])
                scores.append(data['score'])
                labels.append(data['label'])
                print(f"CoordinatorAgent: Collected fused box {data} from blackboard")

        print(f"CoordinatorAgent: Posting final fused boxes: {fused_boxes}, scores: {scores}, labels: {labels}")
        self.blackboard.post("final_fused_boxes", {"boxes": fused_boxes, "scores": scores, "labels": labels})
