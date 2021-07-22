import ark_tweet.CMUTweetTagger as ct
import re
import numpy as np
import collections
from fasttext import load_model
import epitran
import panphon
import csv
from gensim.models import KeyedVectors


def pos_tags(X, separator="separateplease"):

    # X = list of tokenized texts

    flat_X = []

    for sample in X:
        for token in sample:
            flat_X.append(token)
        flat_X.append(separator)

    flat_X_tagged = ct.runtagger_parse(flat_X)

    X_pos = []
    sample_pos = []

    for flat_sample in flat_X_tagged:
        token_tuple = flat_sample[0]
        token = token_tuple[0]
        pos = token_tuple[1]

        if token == separator:
            X_pos.append(sample_pos)
            sample_pos = []
        else:
            sample_pos.append(pos)

    return X_pos


def make_sure_length_alignment(X, Y, pos):
    for sample_X, sample_Y, sample_pos in zip(X, Y, pos):
        assert len(sample_X) == len(sample_Y)
        assert len(sample_X) == len(sample_pos)


def prepare_mixed_case(X, Y, pos):
    mixed_X = []
    mixed_Y = []
    mixed_pos = []

    for sample_X, sample_Y, sample_pos in zip(X, Y, pos):

        mixed_X.append(sample_X)
        mixed_Y.append(sample_Y)
        mixed_pos.append(sample_pos)

        temp = " ".join(sample_X)
        temp_lower = temp.lower()

        if temp != temp_lower:
            sample_X_low = temp_lower.split(" ")
            mixed_X.append(sample_X_low)
            mixed_Y.append(sample_Y)
            mixed_pos.append(sample_pos)

    return mixed_X, mixed_Y, mixed_pos


def preprocess_token(word):

    if 'https://' in word or 'http://' in word or 'www.' in word:
        return "http"
    elif '@' in word[0] and word != '@':
        return "@user"
    else:
        return word


def preprocess(X):

    new_X = []

    for sample in X:
        new_sample = [preprocess_token(token) for token in sample]
        new_X.append(new_sample)

    return new_X


def reorder(items, idx):
    return [items[i] for i in idx]


def create_vocab2glove(embedding_path="../embeddings/glove/glove.twitter.27B.200d.txt", word_vec_dim=200):

    vocab2glove = {}
    with open(embedding_path) as infile:
        for line in infile:
            row = line.strip().split(' ')
            word = row[0]
            if word not in vocab2glove:
                vec = np.asarray(row[1:], np.float32)
                if len(vec) == word_vec_dim:
                    vocab2glove[word] = vec

    # print('Embedding Loaded.')

    return vocab2glove


def create_vocab2w2v(embedding_path="../embeddings/word2vec/word2vec_twitter_tokens.bin"):

    vocab2w2v = KeyedVectors.load_word2vec_format(
        "../embeddings/word2vec/word2vec_twitter_tokens.bin", unicode_errors='ignore', binary=True)  # C bin format
    return vocab2w2v


def load_glove(word, vocab2glove, word_vec_dim=200):

    if word == "<pad>":
        return np.zeros((word_vec_dim), np.float32)
    elif word == "<unk>":
        return np.random.randn(word_vec_dim)
    else:
        return vocab2glove[word]


def load_w2v(word, vocab2w2v, word_vec_dim=400):

    if word == "<pad>":
        return np.zeros((word_vec_dim), np.float32)
    elif word == "<unk>":
        return np.random.randn(word_vec_dim)
    else:
        return vocab2w2v[word]


def load_fasttext(word, ft_model, word_vec_dim=300):

    if word == "<pad>":
        return np.zeros((word_vec_dim), np.float32)
    elif word == "<unk>":
        return np.random.randn(word_vec_dim)
    else:
        return ft_model.get_word_vector(word)


def ipa2vec(ipa_token, ipa2idx, max_char_len=15):

    pad = 0
    ipa_vec = [ipa2idx.get(ipa, pad) for ipa in ipa_token]

    while len(ipa_vec) < max_char_len:
        ipa_vec.append(pad)

    if len(ipa_vec) > max_char_len:
        ipa_vec = ipa_vec[0:max_char_len]

    return ipa_vec


def ipa2phono(ipa_token, phono_feats, ft, max_char_len=15):

    phono_vec = ft.word_array(phono_feats, ''.join(ipa_token)).tolist()
    pad = [0]*len(phono_feats)

    while len(phono_vec) < max_char_len:
        phono_vec.append(pad)

    if len(phono_vec) > max_char_len:
        phono_vec = phono_vec[0:max_char_len]

    return phono_vec


def prepare_pos_vocab():
    tweet_pos_vocab = ['N', 'O', 'S', '^', 'Z', 'V', 'L', 'M', 'A', 'R', '!',
                       'D', 'P', '&', 'T', 'X', 'Y', '~', 'U', 'E', '$', ',', 'G', '@', '#']

    pos2idx = {}
    for tag in tweet_pos_vocab:
        pos2idx[tag] = len(pos2idx)

    return pos2idx


def prepare_char_vocab(data_list, max_char_len=15):

    epi = epitran.Epitran('eng-Latn')
    ft = panphon.FeatureTable()

    phono_feats = ['syl', 'son', 'cons', 'cont', 'delrel',
                   'lat', 'nas', 'strid', 'voi', 'sg', 'cg',
                   'ant', 'cor', 'distr', 'lab', 'hi', 'lo',
                   'back', 'round', 'velaric', 'tense', 'long']
    ipa2idx = {}

    with open('ipa.csv', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            new_ipa = row['IPA']
            if new_ipa not in ipa2idx:
                ipa2idx[new_ipa] = len(ipa2idx)+1

    vocab = {}
    for sample in data_list:
        for token in sample:
            vocab[token] = 1

    vocab2phono = {}
    vocab2ipa = {}

    for token in vocab:
        ipa_token = epi.trans_list(token)

        ipa_vec = ipa2vec(ipa_token, ipa2idx, max_char_len=max_char_len)
        phono_vec = ipa2phono(ipa_token, phono_feats, ft, max_char_len=max_char_len)

        vocab2ipa[token] = ipa_vec
        vocab2phono[token] = phono_vec

    return vocab2ipa, vocab2phono, ipa2idx


def to_ipa_and_phono(X, vocab2ipa, vocab2phono):

    X_phono = []
    X_ipa = []
    for sample in X:
        sample_phono = [vocab2phono[token] for token in sample]
        sample_ipa = [vocab2ipa[token] for token in sample]
        X_phono.append(sample_phono)
        X_ipa.append(sample_ipa)

    return X_ipa, X_phono


def prepare_vocab(data_list, MAX_VOCAB=50000):

    ft_model = load_model("../embeddings/fasttext/crawl-300d-2M-subword.bin")

    special_tags = ["<unk>", "<pad>"]

    vocab2count = {}
    for X in data_list:
        for sample in X:
            for token in sample:
                vocab2count[token] = vocab2count.get(token, 0) + 1

    vocab = []
    count = []

    for key, val in vocab2count.items():
        vocab.append(key)
        count.append(val)

    sorted_idx = np.flip(np.argsort(count), axis=0)

    vocab = reorder(vocab, sorted_idx)
    vocab = special_tags + vocab

    if len(vocab) > MAX_VOCAB:
        vocab = vocab[0:MAX_VOCAB]

    #vocab2glove = create_vocab2glove()
    vocab2w2v = create_vocab2w2v()

    w2v_vocab = [token for token in vocab if token in vocab2w2v or token in special_tags]

    w2v_vocab2idx = {}
    w2v_embeddings = []

    for i, token in enumerate(w2v_vocab):
        w2v_vocab2idx[token] = i
        w2v_embeddings.append(load_w2v(token, vocab2w2v))

    ft_vocab2idx = {}
    ft_embeddings = []

    for i, token in enumerate(vocab):
        ft_vocab2idx[token] = i
        ft_embeddings.append(load_fasttext(token, ft_model))

    ft_embeddings = np.asarray(ft_embeddings, np.float32)
    w2v_embeddings = np.asarray(w2v_embeddings, np.float32)

    return w2v_vocab2idx, w2v_embeddings, ft_vocab2idx, ft_embeddings


def to_vec(sample, vocab2idx):
    return [vocab2idx.get(token, vocab2idx['<unk>']) for token in sample]


def simplify_labels(sample):
    return [label[0] for label in sample]


def labels_to_vec(sample, labels2idx):
    return [labels2idx[label] for label in sample]
