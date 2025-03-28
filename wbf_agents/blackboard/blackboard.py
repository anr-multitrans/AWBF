import threading

class Blackboard:
    def __init__(self):
        self.data = {}
        self.removed_boxes = {}
        self.evolution_log = {}  # ✅ New log to track intermediate versions
        self.lock = threading.Lock()
        self.sync_event = threading.Event()
        self.marked_for_deletion = set()

    def append_evolution(self, bbox_id, evolution_entry):
        """ Stores the intermediate changes in bounding boxes """
        with self.lock:
            if bbox_id not in self.evolution_log:
                self.evolution_log[bbox_id] = []
            self.evolution_log[bbox_id].append(evolution_entry)

    def get_evolution_log(self):
        """ Retrieve the full evolution history of bounding boxes """
        with self.lock:
            return self.evolution_log.copy()

    def post(self, key, value):
        with self.lock:
            self.data[key] = value

    def read(self, key):
        with self.lock:
            return self.data.get(key, None)

    def read_all(self):
        with self.lock:
            return self.data.copy()

    def delete(self, key):
        with self.lock:
            if key in self.data:
                self.removed_boxes[key] = self.data[key]  # ✅ Move to removed registry instead of deleting permanently
                del self.data[key]

    def get_removed_boxes(self):
        """ Retrieve all removed bounding boxes for reference. """
        with self.lock:
            return self.removed_boxes.copy()

    def sync_barrier(self):
        """ Ensures all agents complete their step before the next round starts. """
        self.sync_event.set()  
        self.sync_event.clear()

    def mark_for_deletion(self, box_id):
        """Mark a box for deletion (soft delete)."""
        with self.lock:
            self.marked_for_deletion.add(box_id)  # ✅ Instead of deleting, mark it

    def is_marked_for_deletion(self, box_id):
        """Check if a bounding box has been marked for deletion."""
        with self.lock:
            return box_id in self.marked_for_deletion

    def delete_marked_boxes(self):
        """Delete all boxes that were marked for deletion at the end of a round."""
        with self.lock:
            for box_id in self.marked_for_deletion:
                if box_id in self.data:
                    del self.data[box_id]
                    print(f"🗑️ Deleted box {box_id} after all agents finished.")
            self.marked_for_deletion.clear()  # ✅ Reset the deletion list after cleanup
