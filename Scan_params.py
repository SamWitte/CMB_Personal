import os
import numpy as np
import sys
import fileinput


rangeList = [(0.01, 0.07), (0.2, 0.32), (60., 80.), (-9.5, -7.5), (0.93, 1.), (6., 18.)]

Ob_L = [0.04]
Oc_L = [0.26]
H0_L = [67.]
Neff_L = [3.04] # Fix This For Now!
As_L = [3e-9]
Ns_L = [0.96]
z_reL = [10]

def generate_samples(N=1000):
    range_list = np.zeros(6)
    min_list = np.zeros(6)
    for i in range(len(rangeList)):
        range_list[i] = rangeList[i][1] - rangeList[i][0]
        min_list[i] = rangeList[i][0]
    samples = [np.random.rand(len(range_list))*range_list + min_list for x in range(N)]
    return samples


def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = replaceExp
        sys.stdout.write(line)

launch_file = 'clusterRunner.py'

sample_list = generate_samples()

for sample in sample_list:
    nef = 3.045
    ob, oc, h0, Ass, ns, zre = sample
    As = 10.**Ass
    print('Computing: ', ob, oc, h0, nef, Ass, ns, zre)
    replaceAll(launch_file,"z_reion =", "z_reion = {:.2f} \n".format(zre))
    replaceAll(launch_file,"Om_b_load =", "Om_b_load = {:.4f} \n".format(ob))
    replaceAll(launch_file,"Om_c_load =", "Om_c_load = {:.4f} \n".format(oc))
    replaceAll(launch_file,"HubbleParam =", "HubbleParam = {:.4f} \n".format(h0))
    replaceAll(launch_file,"n_s_index =", "n_s_index = {:.4f} \n".format(ns))
    replaceAll(launch_file,"Neff =", "Neff = {:.4f} \n".format(nef))
    replaceAll(launch_file,"A_s_norm =", "A_s_norm = {:.4e} \n".format(As))
                        
    os.system("python " + launch_file)

print('Done.')
