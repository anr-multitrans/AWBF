import random
import numpy as np
import pickle
from utilities import calculate_iou

class RLBoundingBoxAgent:
    def __init__(self, bbox_id, bbox_data, blackboard, model_reliability=1.0, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.bbox_id = bbox_id
        self.bbox_data = bbox_data
        self.blackboard = blackboard
        self.model_reliability = model_reliability

        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = self.load_q_table()

    def load_q_table(self):
        try:
            with open("q_table.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_q_table(self):
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def get_state(self):
        return tuple(map(int, self.bbox_data["box"]))  # Convert box to discrete state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(["fuse", "suppress", "adjust"])
        else:
            return max(self.q_table.get(state, {}), key=self.q_table.get(state, {}).get, default="fuse")

    def reward_function(self, iou_before, iou_after):
        return iou_after - iou_before  # Reward if IoU improves

    def update_q_table(self, state, action, reward, new_state):
        self.q_table.setdefault(state, {}).setdefault(action, 0)
        best_future_value = max(self.q_table.get(new_state, {}).values(), default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_future_value - self.q_table[state][action])
        self.save_q_table()

    def train(self, episodes=1000):
        print("🔁 Training RL Agent for Fusion Optimization...")
        for episode in range(episodes):
            iou_before = 0.5  # Example starting IoU
            state = self.get_state()
            action = self.choose_action(state)

            # Simulate environment response
            if action == "fuse":
                iou_after = iou_before + 0.2  # Assume IoU increases
            elif action == "suppress":
                iou_after = iou_before - 0.1
            elif action == "adjust":
                iou_after = iou_before + 0.1

            reward = self.reward_function(iou_before, iou_after)
            new_state = self.get_state()
            self.update_q_table(state, action, reward, new_state)

            if episode % 100 == 0:
                print(f"Episode {episode}: IoU {iou_before} → {iou_after}, Action: {action}, Reward: {reward}")

        print("✅ Training Complete! Q-table saved.")

if __name__ == "__main__":
    blackboard = None  # We don't need the blackboard for training, it's simulated
    sample_bbox = {"box": [10, 20, 30, 40], "score": 0.9, "label": "person"}
    
    agent = RLBoundingBoxAgent(0, sample_bbox, blackboard)
    agent.train(episodes=1000)
