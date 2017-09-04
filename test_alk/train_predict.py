#!/usr/bin/python
# -*- coding: utf-8 -*-

import fasttext

params = {
    'input_file': 'train.txt',
    'output': 'model',
    'label_prefix': '__label__',
    'lr': 1,
    'lr_update_rate': 100,
    'dim': 100,
    'ws': 5,
    'epoch': 50,
    'min_count': 1,
    'neg': 5,
    'word_ngrams': 2,
    'loss': 'hs',
    'bucket': 2000000,
    'minn': 0,
    'maxn': 0,
    'thread': 7,
    't': 0,
    'silent': 1,
    'encoding': 'utf-8',
    'save_softmax': 1,
}
classifier = fasttext.supervised(**params)

result = classifier.test('test.txt')

print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)


classifier.predict(['Fraise Pago'], k=1)
classifier.predict_proba(['Fraise Pago', 'houblon blonde 4°'], k=5)


classifier = fasttext.load_model('model.bin', label_prefix='__label__')
