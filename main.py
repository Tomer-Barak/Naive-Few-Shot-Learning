import copy
import joblib
import matplotlib
import numpy as np
import scipy
from torch.utils.data import DataLoader
import pickle, itertools, time, sys
import training, networks
import Sequential_RPMs as env
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.models as models
import platform


def create_net(HP):
    e_net = networks.old_Z(HP).double()
    t_net = networks.T(HP).double()
    if HP['GPU']:
        e_net = e_net.to('cuda')
        t_net = t_net.to('cuda')
    return e_net, t_net


def create_test(HP):
    seq_RPM = env.seq_RPMs(HP)
    test_sequence = env.TestSequenceDataSet(seq_RPM.data)
    answers = env.AnswersDataSet(seq_RPM.data, seq_RPM.options)
    return test_sequence, answers


def run_single_test(HP):
    z_net, t_net = create_net(HP)

    # Creates the test and answers
    test_sequence, answers = create_test(HP)

    start_test_time = time.time()

    # Optimize the networks
    z_net, t_net, full_answers_prob = training.optimization(test_sequence, z_net, t_net, HP)

    # Determines the winner by the minimum of the last step's loss function
    answers_dataloader = DataLoader(answers, batch_size=1, shuffle=False)
    answers_prob = training.test(z_net, t_net, answers_dataloader, HP)
    answers_prob = np.array(answers_prob)
    if True in np.isnan(answers_prob):
        winner_index = False
    else:
        winner_index = int(np.argmax(answers_prob))
        chosen_value = answers_prob[winner_index]
        winner_index = int(np.random.choice(np.where(answers_prob == chosen_value)[0]))
    print(f'Time to solve test: {np.round(time.time() - start_test_time, 3)}', flush=True)
    return winner_index


def run_multiple_tests(HP):  # multiple tests of the same sequence properties
    chosen_options = np.zeros((HP['N_tests'], HP['num_of_wrong_answers'] + 1))
    for i in range(HP['N_tests']):
        answer_idx = run_single_test(HP)
        if type(answer_idx) == int:
            chosen_options[i, answer_idx] = 1
        print('Test number', i + 1, flush=True)
        success_rate = np.sum(chosen_options[:, 0]) / (i + 1)
        print('Current success rate:', np.round(success_rate, 3), flush=True)
    total_success_rate = np.mean(chosen_options, axis=0)[0]
    print(total_success_rate, flush=True)
    return total_success_rate


if __name__ == "__main__":
    HP = {
        # Build a test by specifying the feature rules:
        # 0-constant feature; 1-random feature ; 2-predictable feature
        'seq_prop': {"color": 1, "position": 0, "size": 2, "shape": 0, "number": 0},

        # Tests hyperparameters
        'grid_size': 100, 'seq_length': 6, 'num_of_wrong_answers': 3,

        # Calculate on a GPU?
        'GPU': True,

        # Optimization hyperparameters
        'lr': 4e-4, 'Z_dim': 1,

        # Number of tests (iterations)
        'N_tests': 500}

    run_multiple_tests(HP)
