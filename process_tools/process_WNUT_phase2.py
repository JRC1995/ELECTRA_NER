from process_utils import *
import json
import pickle

save_common_path = "../processed_data/WNUT_2017/vocab_and_embd.pkl"

save_inter_train_path = "../processed_data/WNUT_2017/train_data_intermediate.json"
save_inter_train_mixed_case_path = "../processed_data/WNUT_2017/train_mixed_data_intermediate.json"
save_inter_dev_path = "../processed_data/WNUT_2017/dev_data_intermediate.json"
save_inter_test_path = "../processed_data/WNUT_2017/test_data_intermediate.json"

save_train_path = "../processed_data/WNUT_2017/train_data.json"
save_train_mixed_case_path = "../processed_data/WNUT_2017/train_mixed_data.json"
save_dev_path = "../processed_data/WNUT_2017/dev_data.json"
save_test_path = "../processed_data/WNUT_2017/test_data.json"

with open(save_common_path, "rb") as fp:
    data = pickle.load(fp)
    w2v_vocab2idx = data['w2v_vocab2idx']
    ft_vocab2idx = data['ft_vocab2idx']
    w2v_embeddings = data['w2v_embeddings']
    ft_embeddings = data['ft_embeddings']
    labels2idx = data['labels2idx']

with open(save_inter_train_path, "r") as fp:
    data = json.load(fp)
    train_X = data["sequence"]
    train_Y = data["labels"]
    train_pos = data["pos_tags"]

with open(save_inter_train_mixed_case_path, "r") as fp:
    data = json.load(fp)
    train_mixed_X = data["sequence"]
    train_mixed_Y = data["labels"]
    train_mixed_pos = data["pos_tags"]

with open(save_inter_dev_path, "r") as fp:
    data = json.load(fp)
    dev_X = data["sequence"]
    dev_Y = data["labels"]
    dev_pos = data["pos_tags"]

with open(save_inter_test_path, "r") as fp:
    data = json.load(fp)
    test_X = data["sequence"]
    test_Y = data["labels"]
    test_pos = data["pos_tags"]

train_segment_Y = [simplify_labels(sample_label) for sample_label in train_Y]
dev_segment_Y = [simplify_labels(sample_label) for sample_label in dev_Y]
test_segment_Y = [simplify_labels(sample_label) for sample_label in test_Y]
train_mixed_segment_Y = [simplify_labels(sample_label) for sample_label in train_mixed_Y]


print("hello6")
segment_labels2idx = {'O': 0, 'B': 1, 'I': 2}
train_segment_Y = [labels_to_vec(sample, segment_labels2idx) for sample in train_segment_Y]
dev_segment_Y = [labels_to_vec(sample, segment_labels2idx) for sample in dev_segment_Y]
test_segment_Y = [labels_to_vec(sample, segment_labels2idx) for sample in test_segment_Y]
train_mixed_segment_Y = [labels_to_vec(sample, segment_labels2idx)
                         for sample in train_mixed_segment_Y]

print("hello7")
train_Y = [labels_to_vec(sample, labels2idx) for sample in train_Y]
dev_Y = [labels_to_vec(sample, labels2idx) for sample in dev_Y]
test_Y = [labels_to_vec(sample, labels2idx) for sample in test_Y]
train_mixed_Y = [labels_to_vec(sample, labels2idx) for sample in train_mixed_Y]


pos2idx = prepare_pos_vocab()
train_pos = [labels_to_vec(sample, pos2idx) for sample in train_pos]
dev_pos = [labels_to_vec(sample, pos2idx) for sample in dev_pos]
test_pos = [labels_to_vec(sample, pos2idx) for sample in test_pos]
train_mixed_pos = [labels_to_vec(sample, pos2idx) for sample in train_mixed_pos]


print("hello1")
vocab2ipa, vocab2phono, ipa2idx = prepare_char_vocab(train_X+dev_X+test_X+train_mixed_X)
print("hello2")
train_ipa, train_phono = to_ipa_and_phono(train_X, vocab2ipa, vocab2phono)
print("hello3")
train_mixed_ipa, train_mixed_phono = to_ipa_and_phono(train_mixed_X, vocab2ipa, vocab2phono)
print("hello4")
dev_ipa, dev_phono = to_ipa_and_phono(dev_X, vocab2ipa, vocab2phono)
print("hello5")
test_ipa, test_phono = to_ipa_and_phono(test_X, vocab2ipa, vocab2phono)
print("hello8")

train_w2v = [to_vec(sample, w2v_vocab2idx) for sample in train_X]
train_ft = [to_vec(sample, ft_vocab2idx) for sample in train_X]

train_mixed_w2v = [to_vec(sample, w2v_vocab2idx) for sample in train_mixed_X]
train_mixed_ft = [to_vec(sample, ft_vocab2idx) for sample in train_mixed_X]

test_w2v = [to_vec(sample, w2v_vocab2idx) for sample in test_X]
test_ft = [to_vec(sample, ft_vocab2idx) for sample in test_X]

dev_w2v = [to_vec(sample, w2v_vocab2idx) for sample in dev_X]
dev_ft = [to_vec(sample, ft_vocab2idx) for sample in dev_X]

make_sure_length_alignment(train_X, train_Y, train_pos)
make_sure_length_alignment(dev_X, dev_Y, dev_pos)
make_sure_length_alignment(test_X, test_Y, test_pos)


with open(save_common_path, "wb") as fp:
    data = {}
    data['w2v_vocab2idx'] = w2v_vocab2idx
    data['ft_vocab2idx'] = ft_vocab2idx
    data['w2v_embeddings'] = w2v_embeddings
    data['ft_embeddings'] = ft_embeddings
    data['labels2idx'] = labels2idx
    data['segment_labels2idx'] = segment_labels2idx
    data['ipa2idx'] = ipa2idx
    data['pos2idx'] = pos2idx
    pickle.dump(data, fp)

with open(save_train_path, "w") as fp:
    data = {}
    data["sequence"] = train_X
    data["phono_feats"] = train_phono
    data["ipa_feats"] = train_ipa
    data["w2v_feats"] = train_w2v
    data["fasttext_feats"] = train_ft
    data["labels"] = train_Y
    data["segment_labels"] = train_segment_Y
    data["pos_tags"] = train_pos
    json.dump(data, fp)

with open(save_train_mixed_case_path, "w") as fp:
    data = {}
    data["sequence"] = train_mixed_X
    data["phono_feats"] = train_mixed_phono
    data["ipa_feats"] = train_mixed_ipa
    data["w2v_feats"] = train_mixed_w2v
    data["fasttext_feats"] = train_mixed_ft
    data["labels"] = train_mixed_Y
    data["segment_labels"] = train_mixed_segment_Y
    data["pos_tags"] = train_mixed_pos
    json.dump(data, fp)

with open(save_dev_path, "w") as fp:
    data = {}
    data["sequence"] = dev_X
    data["phono_feats"] = dev_phono
    data["ipa_feats"] = dev_ipa
    data["w2v_feats"] = dev_w2v
    data["fasttext_feats"] = dev_ft
    data["labels"] = dev_Y
    data["segment_labels"] = dev_segment_Y
    data["pos_tags"] = dev_pos
    json.dump(data, fp)

with open(save_test_path, "w") as fp:
    data = {}
    data["sequence"] = test_X
    data["phono_feats"] = test_phono
    data["ipa_feats"] = test_ipa
    data["w2v_feats"] = test_w2v
    data["fasttext_feats"] = test_ft
    data["labels"] = test_Y
    data["segment_labels"] = test_segment_Y
    data["pos_tags"] = test_pos
    json.dump(data, fp)
