#!/usr/bin/env python

"""
Usage: python vis_finetune_testloss.py file1 [file2] [file3]
"""

import numpy as np
import re
import click
from matplotlib import pylab as plt
import os
import itertools

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax1_legends = []
    handles = []
    for i, log_file in enumerate(files):
        basename = os.path.basename(log_file)
        ax1_legends.append(basename + ' training loss')
        ax1_legends.append(basename + ' testing loss')
        loss_iterations, losses, test_loss_iterations, test_losses, test_losses_iteration_checkpoints_ind = parse_log(log_file)
        p = disp_results(fig, ax1, ax1, loss_iterations, losses, test_loss_iterations, test_losses, test_losses_iteration_checkpoints_ind, color_ind=i)
        handles += p  # join plot handle

    ax1.legend(flip(handles, 2), flip(ax1_legends, 2), loc=2, ncol=2)
    plt.show()


def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    test_loss_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n(?:.*\n)?.* *loss = (?P<test_loss>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    test_losses = []
    test_loss_iterations = []
    test_losses_iteration_checkpoints_ind = []

    for r in re.findall(test_loss_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1])

        if iteration % 10000 == 0 and iteration > 0:
            test_losses_iteration_checkpoints_ind.append(len(test_loss_iterations))

        test_loss_iterations.append(iteration)
        test_losses.append(accuracy)

    test_loss_iterations = np.array(test_loss_iterations)
    test_losses = np.array(test_losses)

    return loss_iterations, losses, test_loss_iterations, test_losses, test_losses_iteration_checkpoints_ind


def disp_results(fig, ax1, ax2, loss_iterations, losses, test_loss_iterations, test_losses, test_losses_iteration_checkpoints_ind, color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    p1, = ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])
    p2, = ax2.plot(test_loss_iterations, test_losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula], linestyle='--', marker='o')
    #ax2.plot(test_loss_iterations[test_losses_iteration_checkpoints_ind], test_losses[test_losses_iteration_checkpoints_ind], 'o', color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])
    return [p1, p2]

if __name__ == '__main__':
    main()
