import os
import time
import numpy as np
import json
import random
import copy

# generate some integers

# from data_reader import load_data, DataReader

# if __name__=='__main__':
#     ''' Trains model from data '''

#     word_vocab, word_tensors, max_doc_length, label_tensors = \
#         load_data('data', 15, 50)
#     print(word_vocab)
#     print(word_tensors)
#     print(label_tensors)

#     train_reader = DataReader(word_tensors['train'], label_tensors['train'],
#                               20)

#     valid_reader = DataReader(word_tensors['valid'], label_tensors['valid'],
#                               20)

#     test_reader = DataReader(word_tensors['test'], label_tensors['test'],
#                               20)
validate=[]
examples=[]
index=random.sample(range(0, 2742), 243)
with open('data/train.json') as f:
	examples = [json.loads(line) for line in f]
	print(len(examples))
	for value in index:
		print(value)
		validate.append(examples[value])
	index.sort(reverse=True)
	for i in index:
		examples.pop(i)
	print(len(examples))
f.close()
with open("data/train_new.json",'w') as f:
	for row in examples:
		f.write(json.dumps(row, ensure_ascii=False) + "\n")
f.close()

with open("data/valid.json",'w') as f:
	for row in validate:
		f.write(json.dumps(row, ensure_ascii=False) + "\n")
f.close()
