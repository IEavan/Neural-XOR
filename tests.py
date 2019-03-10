import torch
import numpy as np
import tqdm

from network import MLP
from torch.autograd import Variable

N_ITERS = 10000

# Question Two
for n_vectors in [16, 32, 64]:
    for n_hidden in [2, 4, 8]:
        print("Num Vecttors: {}\nHidden Units: {}".format(n_vectors, n_hidden))

        mlp = MLP(n_hidden)

        clean_x = np.random.randint(2, size=(n_vectors, 2))
        y = np.remainder(np.sum(clean_x, axis=1), 2)
        noise = np.random.normal(scale=0.5, size=(n_vectors, 2))
        x = clean_x + noise

        x = Variable(torch.from_numpy(x))
        y = Variable(torch.from_numpy(y))

        optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-4)
        mse_loss =  torch.nn.MSELoss()

        with tqdm.tqdm(total=N_ITERS) as t:
            for epoch in range(N_ITERS):
                y_pred = mlp(x)
                loss = mse_loss(y_pred.squeeze(), y.float())
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss=loss.item())
                t.update()
