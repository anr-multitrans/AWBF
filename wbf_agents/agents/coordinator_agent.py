import networkx as nx

class CoordinatorAgent:
    def __init__(self, blackboard):
        self.blackboard = blackboard
        self.knowledge_graph = self.build_coco_knowledge_graph()
        
        if self.knowledge_graph is not None:
            self.transitive_weights = self.compute_transitive_weights(self.knowledge_graph)
        else:
            self.transitive_weights = {}

    def compute_transitive_weights(self, G):
        """
        Computes transitive relationships dynamically using shortest paths.
        """
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))  
        transitive_weights = {}

        for source in G.nodes:
            transitive_weights[source] = {}
            for target in G.nodes:
                if source == target:
                    continue  

                if G.has_edge(source, target):
                    transitive_weights[source][target] = G[source][target]["weight"]
                else:
                    try:
                        path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")
                        weight = 1.0
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            weight *= G[u][v]["weight"]

                        transitive_weights[source][target] = weight if weight > 0 else 0.0  
                    except nx.NetworkXNoPath:
                        transitive_weights[source][target] = 0.0  

        return transitive_weights

    def review_and_decide(self):
        """
        Collects fused bounding boxes, applies contextual reasoning, and removes redundant boxes.
        """
        fused_boxes = [
            data for key, data in self.blackboard.read_all().items()
            if isinstance(data, dict) and "box" in data and "ancestry" in data
        ]

        referenced_boxes = set()
        for box in fused_boxes:
            referenced_boxes.update(box["ancestry"])

        final_boxes = []
        for box in fused_boxes:
            new_box = box.copy()
            boost_factor = self.get_dynamic_boost(box)
            if boost_factor > 0:
                new_box["score"] = self.boost_confidence(box["score"], boost_factor)
            final_boxes.append(new_box)

        self.blackboard.post(
            "final_fused_boxes",
            {"boxes": [b["box"] for b in final_boxes], 
             "scores": [b["score"] for b in final_boxes], 
             "labels": [b["label"] for b in final_boxes]}
        )

        # ✅ Remove all referenced bounding boxes
        for key in list(self.blackboard.read_all().keys()):
            # Remove all non-final bounding boxes and intermediate fused boxes
            if key != "final_fused_boxes":  
                self.blackboard.delete(key)
                print(f"🗑️ Removed {key} from blackboard after final fusion.")

        # ✅ Print removed boxes history
        removed_boxes = self.blackboard.get_removed_boxes()
        print(f"🗂️ Removed Bounding Boxes History: {removed_boxes}")

        # ✅ Print evolution history of bounding boxes
        evolution_log = self.blackboard.get_evolution_log()
        print(f"📈 Bounding Box Evolution Log: {evolution_log}")


        

    def get_dynamic_boost(self, box):
        """
        Determines the confidence boost dynamically using transitive relations.
        """
        label = box["label"]
        detected_objects = [b["label"] for b in self.blackboard.read_all().values() if "label" in b]

        max_boost = 0.2  
        min_boost = 0.0  
        boost_factor = 0.0  

        for neighbor in detected_objects:
            if label in self.transitive_weights and neighbor in self.transitive_weights[label]:
                weight = self.transitive_weights[label][neighbor]
                if weight > 0:
                    boost_factor = max(boost_factor, min_boost + (max_boost - min_boost) * weight)  

        return boost_factor  

    def boost_confidence(self, current_score, boost_factor):
        """
        Boosts confidence dynamically while ensuring it does not exceed 1.0.
        """
        return min(1.0, current_score + (1.0 - current_score) * boost_factor)


    def build_coco_knowledge_graph(self):
        """
        Builds a weighted knowledge graph for COCO dataset categories.
        Strong relationships get high weights (0.9 - 1.0).
        Weaker relationships are transitive (0.5 - 0.8).
        Explicit non-relations are blocked (0.0).
        """
        G = nx.Graph()

        # 🚗 Transportation (Strong)
        G.add_edge("car", "truck", weight=0.9)
        G.add_edge("car", "bus", weight=0.9)
        G.add_edge("bicycle", "motorcycle", weight=0.9)
        G.add_edge("bicycle", "person", weight=1.0)
        G.add_edge("motorcycle", "person", weight=0.8)
        G.add_edge("car", "road", weight=1.0)
        G.add_edge("truck", "road", weight=1.0)
        G.add_edge("bus", "road", weight=1.0)
        G.add_edge("traffic light", "road", weight=0.9)
        G.add_edge("stop sign", "road", weight=0.9)

        # 👨‍👩‍👧‍👦 People & Clothing
        G.add_edge("person", "backpack", weight=0.9)
        G.add_edge("person", "handbag", weight=0.9)
        G.add_edge("person", "umbrella", weight=0.8)
        G.add_edge("person", "tie", weight=0.7)
        G.add_edge("person", "hat", weight=0.8)

        # 🏡 Indoor Objects
        G.add_edge("chair", "table", weight=0.9)
        G.add_edge("couch", "table", weight=0.8)
        G.add_edge("bed", "pillow", weight=0.9)
        G.add_edge("book", "table", weight=0.8)
        G.add_edge("clock", "table", weight=0.7)
        G.add_edge("vase", "table", weight=0.8)

        # 🌳 Outdoor & Nature
        G.add_edge("tree", "grass", weight=0.9)
        G.add_edge("flower", "grass", weight=0.8)
        G.add_edge("bird", "tree", weight=0.9)
        G.add_edge("dog", "person", weight=0.9)
        G.add_edge("cat", "person", weight=0.8)
        G.add_edge("dog", "grass", weight=0.7)
        G.add_edge("cat", "grass", weight=0.7)

        # 🍔 Food (Strong)
        G.add_edge("pizza", "hot dog", weight=0.8)
        G.add_edge("pizza", "sandwich", weight=0.8)
        G.add_edge("banana", "apple", weight=0.9)
        G.add_edge("bottle", "cup", weight=0.9)

        # 🚫 Explicitly Blocked Relations (No Connection)
        G.add_edge("guitar", "pizza", weight=0.0)
        G.add_edge("airplane", "car", weight=0.0)
        G.add_edge("bicycle", "sandwich", weight=0.0)
        G.add_edge("dog", "stop sign", weight=0.0)
        G.add_edge("cat", "traffic light", weight=0.0)

        return G

    def build_knowledge_graph(self):
        """
        Constructs a knowledge graph with direct relations only.
        Indirect relations will be derived dynamically using shortest path calculations.
        """
        G = nx.Graph()

        # Define only direct relationships
        G.add_edge("bicycle", "person", weight=0.9)  
        G.add_edge("person", "car", weight=0.8)  
        G.add_edge("car", "road", weight=1.0)
        G.add_edge("dog", "person", weight=0.7)

        # Explicitly block some relationships by setting weight = 0
        G.add_edge("guitar", "pizza", weight=0.0)

        return G
