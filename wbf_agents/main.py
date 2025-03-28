import numpy as np
from blackboard import Blackboard
from agents.bounding_box_agent import BoundingBoxAgent
from agents.model_specific_agent import ModelSpecificAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.data_processing_agent import DataProcessingAgent
from agents.rl_bounding_box_agent import RLBoundingBoxAgent  # Import RL-based agent

def test_bidding():
    # Create a blackboard for communication
    blackboard = Blackboard()

    # Example bounding boxes (x1, y1, x2, y2) with confidence scores
    #bbox_data_1 = {"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "label": "car"}
    #bbox_data_2 = {"box": [0.12, 0.22, 0.32, 0.42], "score": 0.85, "label": "car"}

    bbox_data_1 = {"box": [0.1, 0.2, 0.35, 0.45], "score": 0.75, "label": "person"}
    bbox_data_2 = {"box": [0.3, 0.35, 0.5, 0.55], "score": 0.72, "label": "person"}

    # Create two agents (Multi-Round Mode)
    agent1 = BoundingBoxAgent("box1", bbox_data_1, blackboard, model_reliability=1.0, use_multi_round=True)
    agent2 = BoundingBoxAgent("box2", bbox_data_2, blackboard, model_reliability=0.9, use_multi_round=True)

    # Agents analyze and negotiate
    agent1.analyze_and_propose()
    agent2.analyze_and_propose()

    # Coordinator Agent makes the final decision
    coordinator = CoordinatorAgent(blackboard)
    coordinator.review_and_decide()

    # Read the final fusion result
    #print("Multi-Round Negotiation Final Result:", blackboard.read("final_fused_boxes"))   
    final_result = blackboard.read("final_fused_boxes")
    print("✅ Cleaned Final Fused Boxes:", final_result)

    # ✅ Verify cleanup (debugging)
    print("🗑️ Remaining Blackboard Data:", blackboard.read_all())  # Should contain only fused boxes
 
def test_multi_round_negotiation():
    blackboard = Blackboard()
#86*118,538*434

#64*209, 578*434
    bbox_data_1 = {"box": [86, 118, 538, 434], "score": 0.8, "label": "motorcycle"}
    bbox_data_2 = {"box": [64, 209, 578, 434], "score": 0.7, "label": "motorcycle"}

    agent1 = BoundingBoxAgent("box1", bbox_data_1, blackboard, model_reliability=1.0, use_multi_round=True, debug=True)
    agent2 = BoundingBoxAgent("box2", bbox_data_2, blackboard, model_reliability=0.9, use_multi_round=True, debug=True)

    agent1.analyze_and_propose()
    agent2.analyze_and_propose()

    coordinator = CoordinatorAgent(blackboard)
    coordinator.review_and_decide()

    final_result = blackboard.read("final_fused_boxes")
    print("✅ Final Fused Boxes After Cleanup:", final_result)

    remaining_data = blackboard.read_all()
    if list(remaining_data.keys()) == ["final_fused_boxes"]:
        print("🎉 Cleanup successful! Only final fused boxes remain.")
    else:
        print("⚠️ Warning: Unexpected data in blackboard!", remaining_data)

    # ✅ Print removed boxes registry
    print("🗂️ Removed Boxes Registry:", blackboard.get_removed_boxes())


    

def test_negotiation():
    print("\n🔍 Testing Negotiation-Based Fusion\n")

    blackboard = Blackboard()
    data_processor = DataProcessingAgent()

    bbox_data = [
        {"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "label": "person"},
        {"box": [0.11, 0.21, 0.31, 0.41], "score": 0.85, "label": "person"},
        {"box": [0.5, 0.5, 0.6, 0.6], "score": 0.7, "label": "car"}
    ]

    bb_agents = [BoundingBoxAgent(i, data, blackboard, model_reliability=1.0) for i, data in enumerate(bbox_data)]
    model_agent = ModelSpecificAgent('modelA', blackboard)
    coordinator = CoordinatorAgent(blackboard)

    for agent in bb_agents:
        agent.analyze_and_propose()

    model_agent.adjust_boxes()
    coordinator.review_and_decide()
    final_fused_boxes = blackboard.read("final_fused_boxes")
    postprocessed_boxes = data_processor.postprocess(final_fused_boxes)

    print("✅ Negotiation-Based Fusion Results:", postprocessed_boxes)

def test_rl():
    print("\n🧠 Testing RL-Based Fusion\n")

    blackboard = Blackboard()
    data_processor = DataProcessingAgent()

    bbox_data = [
        {"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "label": "bicycle"},
        {"box": [0.11, 0.21, 0.31, 0.41], "score": 0.85, "label": "bicycle"},
        {"box": [0.5, 0.5, 0.6, 0.6], "score": 0.7, "label": "tree"}
    ]

    rl_agents = [RLBoundingBoxAgent(i, data, blackboard, model_reliability=1.0) for i, data in enumerate(bbox_data)]
    model_agent = ModelSpecificAgent('modelA', blackboard)
    coordinator = CoordinatorAgent(blackboard)

    for agent in rl_agents:
        agent.analyze_and_propose()

    model_agent.adjust_boxes()
    coordinator.review_and_decide()
    final_fused_boxes = blackboard.read("final_fused_boxes")
    postprocessed_boxes = data_processor.postprocess(final_fused_boxes)

    print("✅ RL-Based Fusion Results:", postprocessed_boxes)

if __name__ == "__main__":
    #test_bidding()
    test_multi_round_negotiation()
    #test_negotiation()
    #test_rl()
