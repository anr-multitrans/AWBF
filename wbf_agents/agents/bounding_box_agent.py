import numpy as np
from utils.utilities import calculate_iou
import matplotlib.pyplot as plt

class BoundingBoxAgent:
    def __init__(self, bbox_id, bbox_data, blackboard, model_reliability=1.0, use_multi_round=True, debug=False):
        self.bbox_id = bbox_id
        self.bbox_data = bbox_data
        self.blackboard = blackboard
        self.model_reliability = model_reliability  # Factor for negotiation
        self.use_multi_round = use_multi_round  # Allow switching between single-shot and multi-round
        self.debug=debug

    def analyze_and_propose(self):

        if isinstance(self.bbox_data, list):
            self.bbox_data = self.bbox_data[0] 

        if isinstance(self.bbox_data, (list, np.ndarray)):
            self.bbox_data = {
                'box': self.bbox_data[:4], 
                'score': self.bbox_data[4], 
                'label': self.bbox_data[5]
            }
        overlaps = []
        for key, data in self.blackboard.read_all().items():
            if isinstance(data, list):
                data = data[0]
            print(f"🔍 Debug: data = {data}\n🔍 Debug: bbox_data = {self.bbox_data}")
            
            if not isinstance(data, dict) or 'box' not in data:
                print(f"⚠️ Warning: Skipping key {key} because it lacks 'box' → {data}")
                continue

            if key != self.bbox_id and self.check_overlap(self.bbox_data['box'], data['box']):
                overlaps.append((key, data))

        if overlaps:
            if self.use_multi_round:
                self.multi_round_negotiate_fusion()
            else:
                self.bid_for_fusion(overlaps)
        else:
            print(f"📡 Agent {self.bbox_id} posting box {self.bbox_data}")
            self.blackboard.post(f"{self.bbox_id}", self.bbox_data)

    def check_overlap(self, box1, box2, iou_threshold=0.5):
        return calculate_iou(box1, box2) > iou_threshold

    def bid_for_fusion(self, overlapping_boxes):
        alpha, beta, gamma = 0.5, 0.3, 0.2  # Weighting factors
        proposals = []

        for other_id, data in overlapping_boxes:
            iou = calculate_iou(self.bbox_data['box'], data['box'])
            utility = alpha * self.bbox_data['score'] + beta * iou + gamma * self.model_reliability
            proposals.append((other_id, data, utility))

        if not proposals:
            print(f"📡 Agent {self.bbox_id} posting box {self.bbox_data}")
            self.blackboard.post(f"{self.bbox_id}", self.bbox_data)  # Post original box if no valid fusion
            return

        best_proposal = max(proposals, key=lambda x: x[2])  
        self.fuse_boxes([best_proposal])

    def multi_round_negotiate_fusion(self,  max_rounds=5):
        """
        Implements multi-round negotiation where agents refine their bids.
        Ensures that each agent reads the latest blackboard updates before making its move.
        """
        alpha, beta, gamma = 0.5, 0.3, 0.2
        max_confidence_boost = 1.05
        iou_log = []
        evolution_log = []

        last_best_utility = None

        for round_num in range(max_rounds):
            if self.blackboard.is_marked_for_deletion(self.bbox_id):
                continue
            print(f"\n🔄 Round {round_num+1} Start")

            proposals = []
            round_data = []

            # ✅ Always read the latest bounding boxes from the blackboard
            all_boxes = self.blackboard.read_all()
            overlapping_boxes = [
                (key, data) for key, data in all_boxes.items()                
                if 'box' in data and key != self.bbox_id and self.check_overlap(self.bbox_data['box'], data['box'])
            ]

            if not overlapping_boxes:
                print("⚠️ No overlapping boxes found. Posting original box.")
                print(f"📡 Agent {self.bbox_id} posting box {self.bbox_data}")
                self.blackboard.post(f"{self.bbox_id}", self.bbox_data)
                return

            # 📌 Agents analyze new data from the blackboard before making their move
            to_fuse = []  # ✅ Track boxes to fuse
            to_remove = []  # ✅ Track boxes to remove after fusion

            for other_id, data in overlapping_boxes:
                iou = calculate_iou(self.bbox_data['box'], data['box'])

                if iou > 0.99:  # ✅ Identical bounding box detected
                    print(f"⚠️ IoU ≈ 1.0 → Removing redundant box {other_id} from blackboard")
                    to_fuse.append( data)  # ✅ Mark for fusion
                    to_remove.append(other_id)
                    self.blackboard.mark_for_deletion(other_id)
                    continue  # ✅ Skip this box in negotiation

                # ✅ Only add unique boxes to negotiation
                utility = alpha * self.bbox_data['score'] + beta * iou + gamma * self.model_reliability
                proposals.append((other_id, data, utility, iou))
                round_data.append((data['box'], data['score'], iou))

            # ✅ Delete redundant bounding boxes from the blackboard
            if to_fuse:
                all_boxes = [self.bbox_data] + to_fuse  # ✅ Include current box in fusion
                fused_box = {
                    'box': np.mean([b['box'] for b in all_boxes if isinstance(b, dict)], axis=0).tolist(),
                    'score': np.mean([b['score'] for b in all_boxes]),  # ✅ Average confidence score
                    'label': self.bbox_data['label'],  # ✅ Keep the same label
                    'ancestry': [self.bbox_id] + [b['ancestry'] if 'ancestry' in b else other_id for other_id, b in enumerate(to_remove)]
                }

                # ✅ Post the fused box and exit early to prevent unnecessary negotiation
                print(f"📡 Posting early fused box {fused_box} instead of individual boxes.")
                self.blackboard.post(f"fused_{self.bbox_id}", fused_box)
                return  # ✅ Skip further negotiation since fusion already happened
            
            #for id_b in to_remove:
            #    self.blackboard.mark_for_deletion(id_b)
            if proposals:
                best_proposal = max(proposals, key=lambda x: x[2])
                best_utility = best_proposal[2]
                best_iou = best_proposal[3]
                iou_log.append(best_iou)

                print(f"  ✅ Best Proposal (Round {round_num+1}): Utility={best_utility:.3f}, IoU={best_iou:.3f}")

                # 🔄 Agents dynamically adjust their own box based on the latest updates
                if last_best_utility is not None and best_utility <= last_best_utility:
                    print(f"⚠️ Utility did not improve. Adjusting my own box.")

                    bb1 = self.bbox_data["box"]
                    best_box = best_proposal[1]["box"]

                    # 🔄 Move **my own box** toward the best competing proposal
                    # First: Store previous version before modification
                    evolution_entry = {
                        "round": round_num + 1,
                        "previous_box": self.bbox_data["box"],
                        "adjusted_box": [(bb1[i] * 0.7 + best_box[i] * 0.3) for i in range(4)],
                        "iou": best_iou,
                        "utility": best_utility
                    }
                    self.blackboard.append_evolution(self.bbox_id, evolution_entry)

                    # Apply adjustment
                    self.bbox_data["box"] = evolution_entry["adjusted_box"]


                    # ✅ Post **only my updated box** to the blackboard
                    print(f"📡 Agent {self.bbox_id} posting box {self.bbox_data}")
                    self.blackboard.post(f"{self.bbox_id}", self.bbox_data)

                last_best_utility = best_utility
                evolution_log.append(round_data)

                # ✅ Ensure each agent sees the blackboard updates **before merging**
                self.blackboard.sync_barrier()  # 🔄 New function to synchronize agent updates

                if best_iou > 0.7:
                    print(f"  ✅ Merging boxes in Round {round_num+1} (IoU reached threshold)")
                    self.fuse_boxes([best_proposal])
                    return
            self.blackboard.sync_barrier()  # 🔄 Ensure all agents finish
            self.blackboard.delete_marked_boxes()  # ✅ Now safely remove marked boxes

        print("🔚 Posting final fused box to Blackboard.")
        print(f"📡 Agent {self.bbox_id} posting box {best_proposal[1]}")

        fused_box_key = f'fused_{self.bbox_id}'
        self.blackboard.post(fused_box_key, best_proposal[1])

        # ✅ Remove initial bounding box from blackboard
        for key, data in self.blackboard.read_all().items():
            if isinstance(data, dict) and "box" in data:
                if key != fused_box_key and not key.startswith("final_fused_boxes"):
                    self.blackboard.delete(key)
                    print(f"🗑️ Removed {key} from blackboard after fusion.")
        #self.blackboard.delete(self.bbox_id)
        #print(f"🗑️ Agent {self.bbox_id} removed its initial bounding box after fusion.")

        #self.blackboard.post(f'fused_{self.bbox_id}', best_proposal[1])


    def plot_iou_evolution(self, iou_log):
        """
        Plots IoU changes over negotiation rounds.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(iou_log)), iou_log, marker='o', linestyle='-', color='b', label="IoU Evolution")
        plt.xlabel("Round Number")
        plt.ylabel("IoU Value")
        plt.title("IoU Evolution Over Negotiation Rounds")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_box_evolution(self, evolution_log):
        """
        Plots bounding box movement across rounds.
        """
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for round_num, round_data in enumerate(evolution_log):
            for box, score, iou in round_data:
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=colors[round_num % len(colors)], facecolor='none', label=f'Round {round_num}')
                ax.add_patch(rect)
                plt.text(x1, y1, f"Score: {score:.2f}\nIoU: {iou:.2f}", color=colors[round_num % len(colors)], fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Bounding Box Evolution Per Round")
        plt.legend()
        plt.show()

    def fuse_boxes(self, selected_boxes):
        """
        Merges bounding boxes using a weighted average and tracks ancestry.
        """
        ancestor_ids = [other_id for other_id, _, _, _ in selected_boxes] + [self.bbox_id]

        fused_box = {
            'box': np.mean([data['box'] for _, data, _, _ in selected_boxes] + [self.bbox_data['box']], axis=0).tolist(),
            'score': np.mean([data['score'] for _, data, _, _ in selected_boxes] + [self.bbox_data['score']]),
            'label': self.bbox_data['label'],
            'ancestry': ancestor_ids  # ✅ Track ancestry of fusion
        }

        fused_box_key = f'fused_{self.bbox_id}'
        self.blackboard.post(fused_box_key, fused_box)
        print(f"📡 Fused box {fused_box_key} with ancestry {ancestor_ids}")

        # ✅ Move removed bounding boxes to registry instead of deleting them
        for ancestor_id in ancestor_ids:
            if ancestor_id in self.blackboard.read_all():
                self.blackboard.delete(ancestor_id)
                print(f"🗑️ Moved ancestor box {ancestor_id} to removed registry.")
