import torch
import numpy as np
import tqdm
import pandas as pd

import matplotlib.style as mlpstyle
import matplotlib.pyplot as plt

from network import MLP
from torch.autograd import Variable

mlpstyle.use("ggplot")
N_ITERS = 50000


def show_mlp_function(mlp, n_vectors, n_hidden):
    x = np.mgrid[0:1:0.01, 0:1:0.01].reshape(2,-1).T
    x = Variable(torch.from_numpy(x))
    y_pred = mlp(x).detach().numpy().reshape(100, 100)

    plt.matshow(y_pred, origin="upper", cmap="magma")
    plt.xticks(np.arange(0,100,25), np.arange(0,1,0.25))
    plt.yticks(np.arange(0,100,25), np.arange(0,1,0.25))
    plt.title("{} hidden units and {} vectors"
              .format(n_hidden, n_vectors))
    plt.savefig("figs/MLP Function with {} hidden units and {} training vectors"
                .format(n_hidden, n_vectors))
    plt.clf()

def show_mlp_loss(ys, n_vectors, n_hidden):
    xs = list(range(len(ys)))
    plt.plot(xs, ys)
    plt.title("{} hidden units and {} vectors"
              .format(n_hidden, n_vectors))
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("figs/MSE Loss with {} hidden units and {} training vectors"
                .format(n_hidden, n_vectors))
    plt.clf()

def calc_test_acc(mlp, loss_func):
    clean_x = np.random.randint(2, size=(64, 2))
    y = np.remainder(np.sum(clean_x, axis=1), 2)
    noise = np.random.normal(scale=0.5, size=(64, 2))
    x = clean_x + noise

    x = Variable(torch.from_numpy(x))
    y = Variable(torch.from_numpy(y))

    y_pred = mlp(x)
    return loss_func(y_pred.squeeze(), y.float()).tolist()


# Question Two
test_accs = []
mse_loss =  torch.nn.MSELoss()

for n_vectors in [16, 32, 64]:
    clean_x = np.random.randint(2, size=(n_vectors, 2))

    y = np.remainder(np.sum(clean_x, axis=1), 2)
    noise = np.random.normal(scale=0.5, size=(n_vectors, 2))
    x = clean_x + noise

    x = Variable(torch.from_numpy(x))
    y = Variable(torch.from_numpy(y))

    for n_hidden in [2, 4, 8]:
        print("Num Vecttors: {}\nHidden Units: {}".format(n_vectors, n_hidden))

        mlp = MLP(n_hidden)
        optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-1)
        losses = []

        with tqdm.tqdm(total=N_ITERS) as t:
            for epoch in range(N_ITERS):
                optimizer.zero_grad()
                y_pred = mlp(x)
                loss = mse_loss(y_pred.squeeze(), y.float())
                losses.append(loss.tolist())
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=loss.item())
                t.update()

        show_mlp_function(mlp, n_vectors, n_hidden)
        show_mlp_loss(losses, n_vectors, n_hidden)
        test_accs.append([n_vectors, n_hidden, calc_test_acc(mlp, mse_loss)])
test_df = pd.DataFrame(test_accs, columns=["n_vectors", "n_hidden", "test_accuracy"])
print(test_df)
test_df.to_csv("test_accuracies.csv")
