import torch
import torch.autograd as Variable

class MLP(torch.nn.Module):
    def __init__(self, n_hidden):
        super(MLP, self).__init__()

        # Define linear layers
        self.linear1 = torch.nn.Linear(2, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        s = self.sigmoid(x)
        
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
