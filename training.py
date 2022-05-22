import torch
import numpy as np
import scipy as sc
from torch.utils.data import DataLoader
import torch.optim as optim
import itertools as it


def CPCLoss(e_t_0, e0, e1):
    all_states = torch.cat((e0, e1[-1].unsqueeze(0)), dim=0)
    loss = 0
    for i in range(len(e_t_0)):
        loss += -torch.log(torch.exp(-(torch.norm(e_t_0[i] - e1[i]) ** 2)) / torch.sum(
            torch.exp(-(torch.norm(e_t_0[i] - all_states, dim=1) ** 2))))
    loss /= len(e_t_0)
    return loss


def test(e_net, t_net, answers_dataloader, HP):
    e_net.eval()
    t_net.eval()
    options_prediction_error = []
    for batch_index, x in enumerate(answers_dataloader):
        if HP['GPU']:
            x[0] = x[0].to('cuda')
            x[1] = x[1].to('cuda')
        e1 = e_net(x[1])
        e0 = e_net(x[0])
        e_t_0 = t_net(e0)
        L_pred = torch.norm(e_t_0 - e1) ** 2
        options_prediction_error.append(L_pred.item())
    answers_prob = sc.special.softmax(-np.array(options_prediction_error))
    print('Answers probabilities:', np.round(answers_prob, 3))
    return answers_prob


def optimization(test_sequence, e_net, t_net, HP):
    optimizer = optim.RMSprop(filter(lambda h: h.requires_grad,
                                     it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])
    answers_prob = []
    data_loader = DataLoader(test_sequence, batch_size=len(test_sequence), shuffle=False)
    for batch_index, x in enumerate(data_loader):
        optimizer.zero_grad()
        e_net.train()
        t_net.train()
        if HP['GPU']:
            x[0] = x[0].to('cuda')
            x[1] = x[1].to('cuda')
        e0 = e_net(x[0])
        e1 = e_net(x[1])
        e_t_0 = t_net(e0)
        loss = CPCLoss(e_t_0, e0, e1)
        loss.backward()
        optimizer.step()
        print("CPC loss " + str(round(loss.item(), 10)))
    answers_prob = np.array(answers_prob)
    return e_net, t_net, answers_prob
