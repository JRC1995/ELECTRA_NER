
import csv
from itertools import groupby
from process_utils import *
import json
import pickle

train_path = "../data/WNUT_2017/emerging.train.conll"
dev_path = "../data/WNUT_2017/emerging.dev.conll"
test_path = "../data/WNUT_2017/emerging.test.conll"

save_common_path = "../processed_data/WNUT_2017/vocab_and_embd.pkl"

save_train_path = "../processed_data/WNUT_2017/train_data_intermediate.json"
save_train_mixed_case_path = "../processed_data/WNUT_2017/train_mixed_data_intermediate.json"
save_dev_path = "../processed_data/WNUT_2017/dev_data_intermediate.json"
save_test_path = "../processed_data/WNUT_2017/test_data_intermediate.json"


# This function is from https://github.com/gaguilar/NER-WNUT17/blob/master/common/utilities.py
def read_file(file_location, delimiter='\t'):
    with open(file_location) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        labeled_tokens = [zip(*g) for k, g in groupby(reader,
                                                      lambda x: not [s for s in x if s.strip()]) if not k]
        tokens, labels = zip(*labeled_tokens)
        return [list(t) for t in tokens], [list(l) for l in labels]


train_X, train_Y = read_file(train_path)
dev_X, dev_Y = read_file(dev_path)
test_X, test_Y = read_file(test_path)

train_pos = pos_tags(train_X)
dev_pos = pos_tags(dev_X)
test_pos = pos_tags(test_X)

train_X = preprocess(train_X)
dev_X = preprocess(dev_X)
test_X = preprocess(test_X)

make_sure_length_alignment(train_X, train_Y, train_pos)
make_sure_length_alignment(dev_X, dev_Y, dev_pos)
make_sure_length_alignment(test_X, test_Y, test_pos)


labels2idx = {}
for sample in train_Y + dev_Y + test_Y:
    for label in sample:
        if label not in labels2idx:
            labels2idx[label] = len(labels2idx)

# print(labels2idx)

train_X_mixed, train_Y_mixed, train_pos_mixed = prepare_mixed_case(train_X, train_Y, train_pos)
w2v_vocab2idx, w2v_embeddings, ft_vocab2idx, ft_embeddings = prepare_vocab([train_X_mixed])

with open(save_common_path, "wb") as fp:
    data = {}
    data['w2v_vocab2idx'] = w2v_vocab2idx
    data['ft_vocab2idx'] = ft_vocab2idx
    data['w2v_embeddings'] = w2v_embeddings
    data['ft_embeddings'] = ft_embeddings
    data['labels2idx'] = labels2idx
    pickle.dump(data, fp)

with open(save_train_path, "w") as fp:
    data = {}
    data["sequence"] = train_X
    data["labels"] = train_Y
    data["pos_tags"] = train_pos
    json.dump(data, fp)

with open(save_train_mixed_case_path, "w") as fp:
    data = {}
    data["sequence"] = train_X_mixed
    data["labels"] = train_Y_mixed
    data["pos_tags"] = train_pos_mixed
    json.dump(data, fp)

with open(save_dev_path, "w") as fp:
    data = {}
    data["sequence"] = dev_X
    data["labels"] = dev_Y
    data["pos_tags"] = dev_pos
    json.dump(data, fp)

with open(save_test_path, "w") as fp:
    data = {}
    data["sequence"] = test_X
    data["labels"] = test_Y
    data["pos_tags"] = test_pos
    json.dump(data, fp)
