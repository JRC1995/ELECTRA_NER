import torch as T
#from models.layers.BigTransformers.BERT import BertModel
#import numpy as np
#from configs.WNUT_configs import *
#from transformers import BertTokenizerFast, ElectraTokenizerFast


def pool(list_embeddings, pool_type='mean'):
    if pool_type == 'first':
        return list_embeddings[0]
    elif pool_type == 'mean':
        return T.mean(T.stack(list_embeddings), dim=0)
    elif pool_type == 'max':
        values, _ = T.max(T.stack(list_embeddings), dim=0)
        return values
    elif pool_type == 'min':
        values, _ = T.min(T.stack(list_embeddings), dim=0)
        return values


def word_level_len(sample):
    count = 0
    for tag in sample:
        if tag in ['B', 'S']:
            count += 1
    return count


def word_pool(subtoken_embeddings, word_info, PAD_EMBD, pool_type='mean'):

    pool_type = pool_type.lower()
    if pool_type not in ['mean', 'max', 'first', 'min']:
        raise ValueError("Legal word pool types are 'first', 'mean', 'max', and 'min'.")

    subtoken_embeddings = subtoken_embeddings[:, 1:, :]  # remove CLS

    N, S, D = subtoken_embeddings.size()

    batch_word_embeddings = []

    max_word_len = max([word_level_len(sample) for sample in word_info])

    for b in range(N):
        temp_pool_list = []
        sample_word_embeddings = []
        for i, word_tag in enumerate(word_info[b]):
            if word_tag == 'S':
                sample_word_embeddings.append(subtoken_embeddings[b, i])
            elif word_tag == 'E':
                temp_pool_list.append(subtoken_embeddings[b, i])
                word_embedding = pool(temp_pool_list, pool_type=pool_type)
                sample_word_embeddings.append(word_embedding)
            elif word_tag == 'B' or word_tag == 'I':
                temp_pool_list.append(subtoken_embeddings[b, i])
            elif word_tag == 'SEP' or word_tag == 'PAD':
                pass
        while len(sample_word_embeddings) < max_word_len:
            sample_word_embeddings.append(PAD_EMBD)

        sample_word_embeddings = T.stack(sample_word_embeddings)

        assert sample_word_embeddings.size() == (max_word_len, D)

        batch_word_embeddings.append(sample_word_embeddings)

    batch_word_embeddings = T.stack(batch_word_embeddings)

    assert batch_word_embeddings.size() == (N, max_word_len, D)

    return batch_word_embeddings


"""
stuff = "hi how are you doing mylieage ?"
config = BERT_config()
tokenizer = BertTokenizerFast.from_pretrained(config.embedding_path,
                                              output_hidden_states=True,
                                              output_attentions=False)


CLS = tokenizer.encode(tokenizer.cls_token)[1:-1][0]
SEP = tokenizer.encode(tokenizer.sep_token)[1:-1][0]
PAD = tokenizer.encode(tokenizer.pad_token)[1:-1][0]

tokens = [CLS]
word_info = []
for word in stuff.split(" "):
    subtokens = tokenizer.encode(word)[1:-1]
    print(word)
    print(subtokens)
    tokens += subtokens
    if len(subtokens) == 1:
        word_info.append('S')
    else:
        for i in range(len(subtokens)):
            if i == 0:
                word_info.append('B')
            elif i == len(subtokens)-1:
                word_info.append('E')
            else:
                word_info.append('I')
tokens += [SEP]

print(tokens)
print(word_info)

tokens_embeddings = []
for token in tokens:
    tokens_embeddings.append(np.random.randn(768))

x = np.asarray([tokens_embeddings, tokens_embeddings, np.float32)
x = T.tensor(x).float().to('cuda')

word_info = [word_info, word_info]

y = word_pool(x, word_info, PAD_EMBD=T.zeros(768).float().to('cuda'))

print(y.size())
"""
