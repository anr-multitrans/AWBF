from agents import BoundingBoxAgent, ModelSpecificAgent, CoordinatorAgent, DataProcessingAgent
from blackboard import Blackboard

def main():
    # Initialize the blackboard
    blackboard = Blackboard()

    # Example image data (numpy array)
    image_data = np.random.rand(480, 640, 3)  # Mock image data

    # Initialize data processing agent
    data_processor = DataProcessingAgent()

    # Preprocess image data
    preprocessed_image = data_processor.preprocess(image_data)

    # Example bounding boxes data after model prediction
    bbox_data = [
        {"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "label": 1},
        {"box": [0.15, 0.25, 0.35, 0.45], "score": 0.85, "label": 1},
        {"box": [0.5, 0.5, 0.6, 0.6], "score": 0.7, "label": 2}
    ]

    # Initialize bounding box agents
    bb_agents = [BoundingBoxAgent(i, data, blackboard) for i, data in enumerate(bbox_data)]

    # Initialize model-specific agent
    model_agent = ModelSpecificAgent('modelA', blackboard)

    # Initialize coordinator agent
    coordinator = CoordinatorAgent(blackboard)

    # Simulate the workflow
    for agent in bb_agents:
        agent.analyze_and_propose()

    # Model-specific adjustments
    model_agent.adjust_boxes()

    # Coordinator reviews and finalizes the decisions
    coordinator.review_and_decide()

    # Read the final fused boxes
    final_fused_boxes = blackboard.read("final_fused_boxes")

    # Postprocess the fused boxes
    postprocessed_boxes = data_processor.postprocess(final_fused_boxes)

    print(postprocessed_boxes)

if __name__ == "__main__":
    main()
