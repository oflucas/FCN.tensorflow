#!/bin/bash
nohup python FCN.py --batch_size=5 --learning_rate=1e-5 &> run_train.log &
