#!/bin/bash

python Notebook/rnn_train.py \
    --input_file datasets/shakespeare.txt \
    --name shakespeare \
    --num_steps 50 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --max_steps 20000

