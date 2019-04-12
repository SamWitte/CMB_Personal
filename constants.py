import numpy as np
from scipy.interpolate import interp1d
import fileinput
import sys

GravG = 6.707e-39
rho_critical = 1.0537e-5 # Need to multiply by h^2!
kboltz = 8.617e-5 # ev / K
global_a_list = np.concatenate((np.logspace(np.log10(7e-4), np.log10(2e-3), 80),
                                np.logspace(np.log10(2.02e-3), np.log10(4.5e-2), 50),
                                np.logspace(np.log10(5e-2), -0.001, 300)))


def replaceAll(file, searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = replaceExp
        sys.stdout.write(line)
