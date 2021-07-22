from models.optim.optimizers import *
import torch as T
import numpy as np


def tokenize(batch_sequence,
             tokenizer):

    CLS = tokenizer.encode(tokenizer.cls_token)[1:-1][0]
    SEP = tokenizer.encode(tokenizer.sep_token)[1:-1][0]
    PAD = tokenizer.encode(tokenizer.pad_token)[1:-1][0]

    batch_tokens_idx = []
    batch_word_info = []
    batch_mask = []

    for sample in batch_sequence:
        tokens_idx = [CLS]
        word_info = []
        for token in sample:

            subtokens_idx = tokenizer.encode(token)[1:-1]
            tokens_idx += subtokens_idx
            subtoken_len = len(subtokens_idx)

            if subtoken_len == 0:
                subtoken_len = 1
                tokens_idx += [PAD]
                #print("token", token)
            if subtoken_len == 1:
                word_info.append('S')
            else:
                for i in range(subtoken_len):
                    if i == 0:
                        word_info.append('B')
                    elif i == subtoken_len-1:
                        word_info.append('E')
                    else:
                        word_info.append('I')

        tokens_idx += [SEP]
        word_info.append('SEP')  # will be ignored later

        batch_tokens_idx.append(tokens_idx)
        batch_word_info.append(word_info)
        batch_mask.append([1]*len(tokens_idx))

    max_len = max([len(sample) for sample in batch_tokens_idx])

    new_batch_tokens_idx = []
    new_batch_word_info = []
    new_batch_mask = []

    for tokens_idx, word_info, mask in zip(batch_tokens_idx, batch_word_info, batch_mask):

        while len(tokens_idx) < max_len:
            tokens_idx.append(PAD)
            word_info.append('PAD')
            mask.append(0)

        new_batch_tokens_idx.append(tokens_idx)
        new_batch_word_info.append(word_info)
        new_batch_mask.append(mask)

    # print(batch_sequence)
    # print(new_batch_tokens_idx)
    # print(new_batch_word_info)
    # print(new_batch_mask)

    return new_batch_tokens_idx, new_batch_word_info, new_batch_mask


def tokenize_qa(batch_natural_queries,
                batch_sequence,
                tokenizer):

    CLS = tokenizer.encode(tokenizer.cls_token)[1:-1][0]
    SEP = tokenizer.encode(tokenizer.sep_token)[1:-1][0]
    PAD = tokenizer.encode(tokenizer.pad_token)[1:-1][0]

    batch_tokens_idx = []
    batch_token_type_idx = []
    batch_word_info = []
    batch_mask = []

    for i, sample in enumerate(batch_sequence):

        tokens_idx = [CLS]
        word_info = []
        token_type_ids = [0]

        query_idx = tokenizer.encode(batch_natural_queries[i])[1:]

        tokens_idx += query_idx

        for id in query_idx:
            word_info.append("Q")
            token_type_ids.append(0)

        for token in sample:

            subtokens_idx = tokenizer.encode(token)[1:-1]
            tokens_idx += subtokens_idx
            subtoken_len = len(subtokens_idx)

            if subtoken_len == 0:
                subtoken_len = 1
                tokens_idx += [PAD]
                # print("token", token)

            for i in range(subtoken_len):
                token_type_ids.append(1)

            if subtoken_len == 1:
                word_info.append('S')
            else:
                for i in range(subtoken_len):
                    if i == 0:
                        word_info.append('B')
                    elif i == subtoken_len-1:
                        word_info.append('E')
                    else:
                        word_info.append('I')

        tokens_idx += [SEP]
        word_info.append('SEP')  # will be ignored later
        token_type_ids.append(1)

        assert len(token_type_ids) == len(tokens_idx)

        # print(token_type_ids)

        batch_tokens_idx.append(tokens_idx)
        batch_token_type_idx.append(token_type_ids)
        batch_word_info.append(word_info)
        batch_mask.append([1]*len(tokens_idx))

    max_len = max([len(sample) for sample in batch_tokens_idx])

    new_batch_tokens_idx = []
    new_batch_token_type_idx = []
    new_batch_word_info = []
    new_batch_mask = []

    for tokens_idx, token_type_ids, word_info, mask in zip(batch_tokens_idx, batch_token_type_idx, batch_word_info, batch_mask):

        while len(tokens_idx) < max_len:
            tokens_idx.append(PAD)
            token_type_ids.append(1)
            word_info.append('PAD')
            mask.append(0)

        new_batch_tokens_idx.append(tokens_idx)
        new_batch_token_type_idx.append(token_type_ids)
        new_batch_word_info.append(word_info)
        new_batch_mask.append(mask)

    # print(batch_sequence)
    # print(new_batch_tokens_idx)
    # print(new_batch_word_info)
    # print(new_batch_mask)

    return new_batch_tokens_idx, new_batch_token_type_idx, new_batch_word_info, new_batch_mask


def tokenize_query(batch_natural_queries,
                   tokenizer):

    CLS = tokenizer.encode(tokenizer.cls_token)[1:-1][0]
    SEP = tokenizer.encode(tokenizer.sep_token)[1:-1][0]
    PAD = tokenizer.encode(tokenizer.pad_token)[1:-1][0]

    batch_tokens_idx = []
    batch_mask = []

    for i, sample in enumerate(batch_natural_queries):
        tokens_idx = tokenizer.encode(sample)
        batch_tokens_idx.append(tokens_idx)
        batch_mask.append([1]*len(tokens_idx))

    max_len = max([len(sample) for sample in batch_tokens_idx])

    new_batch_tokens_idx = []
    new_batch_mask = []

    for tokens_idx, mask in zip(batch_tokens_idx, batch_mask):

        while len(tokens_idx) < max_len:
            tokens_idx.append(PAD)
            mask.append(0)

        new_batch_tokens_idx.append(tokens_idx)
        new_batch_mask.append(mask)

    return new_batch_tokens_idx, new_batch_mask


def predict_NER_MRC(model, tokenizer,
                    batch_texts,
                    batch_queries_idx,
                    batch_natural_queries,
                    batch_labels_start,
                    batch_labels_end,
                    batch_segment_labels,
                    batch_mask,
                    device, config, train):

    with T.no_grad():

        if config.use_sequence_label:
            batch_segment_labels = T.tensor(batch_segment_labels).long().to(device)
            batch_labels_start = None
            batch_labels_end = None
        else:
            batch_segment_labels = None
            batch_labels_start = T.tensor(batch_labels_start).long().to(device)
            batch_labels_end = T.tensor(batch_labels_end).long().to(device)

        if config.fine_tune_style is False:

            if config.use_pretrained_query_embedding is True:
                batch_natural_queries_idx, \
                    batch_query_mask = tokenize_query(batch_natural_queries, tokenizer)
                batch_natural_queries_idx = T.tensor(batch_natural_queries_idx).long().to(device)
                batch_query_mask = T.tensor(batch_query_mask).float().to(device)
                batch_queries_idx = None
            else:
                batch_natural_queries_idx = None
                batch_query_mask = None
                batch_queries_idx = T.tensor(batch_queries_idx).long().to(device)

            batch_tokens_idx, batch_word_info, batch_subword_mask = tokenize(batch_texts, tokenizer)

            batch_token_type_idx = None

        else:

            batch_tokens_idx, batch_token_type_idx, batch_word_info, batch_subword_mask = tokenize_qa(batch_natural_queries,
                                                                                                      batch_texts, tokenizer)

            batch_natural_queries_idx = None
            batch_query_mask = None
            batch_queries_idx = None

            batch_token_type_idx = T.tensor(batch_token_type_idx).long().to(device)

        batch_tokens_idx = T.tensor(batch_tokens_idx).long().to(device)
        batch_subword_mask = T.tensor(batch_subword_mask).float().to(device)

        batch_word_mask = T.tensor(batch_mask).float().to(device)

    if train:
        model = model.train()

        predictions, loss = model(x=batch_tokens_idx,
                                  x_token_type_idx=batch_token_type_idx,
                                  subword_mask=batch_subword_mask,
                                  word_mask=batch_word_mask,
                                  word_info=batch_word_info,
                                  x_query=batch_queries_idx,
                                  x_natural_query=batch_natural_queries_idx,
                                  query_mask=batch_query_mask,
                                  y_start=batch_labels_start,
                                  y_end=batch_labels_end,
                                  y=batch_segment_labels)

    else:
        model = model.eval()

        with T.no_grad():
            predictions, loss = model(x=batch_tokens_idx,
                                      subword_mask=batch_subword_mask,
                                      word_mask=batch_word_mask,
                                      word_info=batch_word_info,
                                      x_query=batch_queries_idx,
                                      x_natural_query=batch_natural_queries_idx,
                                      query_mask=batch_query_mask,
                                      y_start=batch_labels_start,
                                      y_end=batch_labels_end,
                                      y=batch_segment_labels)

    return predictions, loss
