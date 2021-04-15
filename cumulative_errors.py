'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://ringnet.is.tue.mpg.de/license). 
Any use not explicitly granted by the LICENSE is prohibited.
Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.
More information about the NoW Challenge is available at https://ringnet.is.tue.mpg.de/challenge.
For comments or questions, please email us at ringnet@tue.mpg.de
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from psbody.mesh import Mesh
import chumpy as ch

def cumulative_error(errors, nbins=100000):
    errors = errors.ravel()
    values, base = np.histogram(errors, bins=nbins) #values, base = np.histogram(1000*errors, bins=nbins)
    cumsum = np.array(np.cumsum(values), dtype=float)
    cumulative = 100.0*cumsum/float(errors.shape[0])
    return (base[:-1], cumulative)

def generating_cumulative_error_plots(method_error_fnames: list, method_identifiers: list, out_fname : str):
    """
    Generate cumulative error plots for a list of errors.
    :param method_error_fnames list of benchmark output files
    :param method_identifiers list of names of methods that created the output files in the order corresdponding to method_error_fnames
    :param out_fname output plot filename
    """
    method_errors = []
    for fname in method_error_fnames:
        method_errors.append(np.load(fname, allow_pickle=True, encoding="latin1").item()['computed_distances'])

    for i in range(len(method_identifiers)):
        print('%s - median: %f, mean: %f, std: %f' % (method_identifiers[i], np.median(np.hstack(method_errors[i])), np.mean(np.hstack(method_errors[i])), np.std(np.hstack(method_errors[i]))))

    cumulative_errors = []
    for error in method_errors:
        cumulative_errors.append(cumulative_error(np.hstack(error)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 8])
    ax.set_xticks(np.arange(0, 8, 1.0))
    ax.set_ylim([0, 100])
    ax.set_yticks(np.arange(0, 101, 20.0))

    for i, method_id in enumerate(method_identifiers):
        plt.plot(cumulative_errors[i][0], cumulative_errors[i][1], label = method_id)

    plt.xlabel('Error [mm]', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    lgd = ax.legend(loc='lower right')
    plt.savefig(out_fname)

if __name__ == '__main__':
    # List of method identifiers, used as method name within the polot
    method_identifiers = []
    # List of paths to the error files (must be of same order than the method identifiers)
    method_error_fnames = []
    # File name of the output error plot
    out_fname = ''

    generating_cumulative_error_plots(method_error_fnames, method_identifiers, out_fname)