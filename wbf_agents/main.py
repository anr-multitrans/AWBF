import numpy as np
from agents import BoundingBoxAgent, ModelSpecificAgent, CoordinatorAgent, DataProcessingAgent
from blackboard import Blackboard

def main():
    blackboard = Blackboard()
    image_data = np.random.rand(480, 640, 3)
    data_processor = DataProcessingAgent()
    preprocessed_image = data_processor.preprocess(image_data)

    bbox_data = [
        {"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "label": 1},
        {"box": [0.11, 0.21, 0.31, 0.41], "score": 0.85, "label": 1},
        {"box": [0.5, 0.5, 0.6, 0.6], "score": 0.7, "label": 2}
    ]

    bb_agents = [BoundingBoxAgent(i, data, blackboard) for i, data in enumerate(bbox_data)]
    model_agent = ModelSpecificAgent('modelA', blackboard)
    coordinator = CoordinatorAgent(blackboard)

    for agent in bb_agents:
        agent.analyze_and_propose()

    model_agent.adjust_boxes()
    coordinator.review_and_decide()
    final_fused_boxes = blackboard.read("final_fused_boxes")
    postprocessed_boxes = data_processor.postprocess(final_fused_boxes)

    print(postprocessed_boxes)

if __name__ == "__main__":
    main()
