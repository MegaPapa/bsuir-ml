
class GradientContainer:

    def __init__(self):
        self.gradient_container = []
        self.loss_container = []
        self.cost_container = []
        self.thetas_container = []

    def get_gradient_container(self):
        return self.gradient_container

    def get_loss_container(self):
        return self.loss_container

    def get_cost_container(self):
        return self.cost_container

    def get_thetas_container(self):
        return self.thetas_container

    def save_snapshot(self, gradient, loss, cost, thetas):
        self.gradient_container.append(gradient)
        self.loss_container.append(loss)
        self.cost_container.append(cost)
        self.thetas_container.append(thetas)
