class LossDisplayer:
    def __init__(self, name_list):
        self.count = 0
        self.name_list = name_list
        self.loss_list = [0] * len(self.name_list)

    def record(self, losses):
        self.count += 1
        for i, loss in enumerate(losses):
            self.loss_list[i] += loss.item()

    def get_avg_losses(self):
        return [loss / self.count for loss in self.loss_list]

    def display(self):
        for i, total_loss in enumerate(self.loss_list):
            avg_loss = total_loss / self.count
            print(f"{self.name_list[i]}: {avg_loss:.4f}   ", end="")

    def reset(self):
        self.count = 0
        self.loss_list = [0] * len(self.name_list)