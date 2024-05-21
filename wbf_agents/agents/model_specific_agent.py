class ModelSpecificAgent:
    def __init__(self, model_type, blackboard):
        self.model_type = model_type
        self.blackboard = blackboard

    def adjust_boxes(self):
        """
        Adjust bounding boxes based on the model type.
        Example adjustments might include scaling coordinates or adjusting scores.
        """
        adjustment_factor = self.get_adjustment_factor()

        for key, data in self.blackboard.read_all().items():
            # Skip already fused boxes
            if key.startswith('fused_'):
                continue
            
            adjusted_box = [coord * adjustment_factor for coord in data['box']]
            adjusted_score = data['score'] * adjustment_factor

            # Update the blackboard with adjusted values
            self.blackboard.post(key, {
                'box': adjusted_box,
                'score': adjusted_score,
                'label': data['label']
            })

    def get_adjustment_factor(self):
        """
        Determine the adjustment factor based on the model type.
        This is a mock-up function and should be tailored to actual model characteristics.
        """
        if self.model_type == 'modelA':
            return 1.1  # Example adjustment factor for model A
        elif self.model_type == 'modelB':
            return 0.9  # Example adjustment factor for model B
        else:
            return 1.0  # Default adjustment factor for other models
