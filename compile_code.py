import os

if not os.path.isdir('OutputFiles'):
    os.mkdir('OutputFiles')
if not os.path.isdir('precomputed'):
    os.mkdir('precomputed')
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
os.system('python setup.py build_ext --inplace')
