class CustomSchedule:
    def __init__(self, d_model=256, warmup_steps=25000, k=10.0):
        """
        Custom learning rate scheduler.
        Args:
            d_model (int): Dimensionality of the model (affects learning rate scaling).
            warmup_steps (int): Number of warmup steps during training.
            k (float): Tunable scalar.
        """
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.k = k
        self.d_model_sqrt = d_model ** -0.5

    def __call__(self, step):
        """
        Calculate the learning rate at a given step.
        Args:
            step (int): Current training step.
        Returns:
            float: Learning rate.
        """
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.k * self.d_model_sqrt * min(arg1, arg2)

