#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import numpy as np

with open('fasttext_dataset.txt', 'r') as f:
    data = f.read().split('\n')

train_size = int(np.floor(0.7*len(data)))
random.shuffle(data)

train_data = data[:train_size]
test_data = data[train_size:]

with open('train.txt', 'w') as tw:
    tw.write('\n'.join(train_data))
with open('test.txt', 'w') as tw:
    tw.write('\n'.join(test_data))
