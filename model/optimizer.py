import torch

class CustomSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model_sqrt = d_model ** -0.5
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        step = max(step, 1)
        return self.d_model_sqrt * min(step ** -0.5, step * (self.warmup_steps ** -1.5))