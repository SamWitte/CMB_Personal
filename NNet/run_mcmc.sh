#!/bin/bash
echo "`date`"
source activate py27
python emcee_runner.py
echo "`date`"