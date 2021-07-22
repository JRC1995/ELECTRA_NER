from models.optim.optimizers import *
import torch as T
import numpy as np




def load_LRangerMod(model, config):

    no_decay = ["embedding", "layernorm", "bias"]

    fine_tune_decay_parameters = [param for name, param in model.named_parameters()
                                  if param.requires_grad and 'bigtransformer' in name.lower() and not any([name.lower() in n for n in no_decay])]

    fine_tune_no_decay_parameters = [param for name, param in model.named_parameters()
                                     if param.requires_grad and 'bigtransformer' in name.lower() and any([name.lower() in n for n in no_decay])]

    decay_parameters = [param for name, param in model.named_parameters()
                        if param.requires_grad and 'bigtransformer' not in name.lower() and not any([name.lower() in n for n in no_decay])]

    no_decay_parameters = [param for name, param in model.named_parameters()
                           if param.requires_grad and'bigtransformer' not in name.lower() and any([name.lower() in n for n in no_decay])]

    optimizer_grouped_parameters = [
        {'params': fine_tune_decay_parameters, 'weight_decay': config.wd, 'lr': config.fine_tune_lr},
        {'params': fine_tune_no_decay_parameters, 'weight_decay': 0.0, 'lr': config.fine_tune_lr},
        {'params': decay_parameters, 'weight_decay': config.wd},
        {'params': no_decay_parameters, 'weight_decay': 0.0},

    ]
    return T.optim.AdamW(optimizer_grouped_parameters,
                         lr=config.lr,
                         weight_decay=config.wd)

    """

    return LRangerMod(optimizer_grouped_parameters,
                      lr=config.lr,
                      weight_decay=config.wd,
                      amsgrad=False,
                      AdaMod=False,
                      warmup=False,
                      IA=False,
                      use_gc=config.use_gc)
    """


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


def predict_NER(model, tokenizer,
                batch_texts,
                batch_w2v_idx, batch_ft_idx,
                batch_pos_idx,
                batch_ipa_idx, batch_phono,
                batch_labels, batch_segment_labels,
                batch_mask,
                device, config, train):

    # print(batch_labels)
    # print(batch_segment_labels)

    #max_len = max([len(text) for text in batch_texts])

    #print("in ml utils: batch_text", max_len)
    #print("in ml utils: batch_labels", np.asarray(batch_labels).shape)

    with T.no_grad():

        if config.use_w2v:
            batch_w2v_idx = T.tensor(batch_w2v_idx).long().to(device)
            batch_ft_idx = None
        elif config.use_fasttext:
            batch_ft_idx = T.tensor(batch_ft_idx).long().to(device)
            batch_w2v_idx = None
        else:
            batch_w2v_idx = None
            batch_ft_idx = None

        if config.use_pos_tags:
            batch_pos_idx = T.tensor(batch_pos_idx).long().to(device)
        else:
            batch_pos_idx = None

        if config.use_char_feats:
            batch_ipa_idx = T.tensor(batch_ipa_idx).long().to(device)
            batch_phono = T.tensor(batch_phono).float().to(device)
        else:
            batch_ipa_idx = None
            batch_phono = None

        if config.use_MTL:
            batch_segment_labels = T.tensor(batch_segment_labels).long().to(device)
        else:
            batch_segment_labels = None

        batch_labels = T.tensor(batch_labels).long().to(device)
        batch_word_mask = T.tensor(batch_mask).float().to(device)

    if tokenizer is not None:
        batch_tokens_idx, batch_word_info, batch_subword_mask = tokenize(batch_texts,
                                                                         tokenizer)

        with T.no_grad():
            batch_tokens_idx = T.tensor(batch_tokens_idx).long().to(device)
            batch_subword_mask = T.tensor(batch_subword_mask).float().to(device)

        if train:
            model = model.train()

            predictions, loss = model(x=batch_tokens_idx,
                                      y=batch_labels,
                                      subword_mask=batch_subword_mask,
                                      word_mask=batch_word_mask,
                                      word_info=batch_word_info,
                                      x_w2v=batch_w2v_idx,
                                      x_fasttext=batch_ft_idx,
                                      x_pos=batch_pos_idx,
                                      x_ipa=batch_ipa_idx,
                                      x_phono=batch_phono)

        else:
            model = model.eval()

            with T.no_grad():
                predictions, loss = model(x=batch_tokens_idx,
                                          y=batch_labels,
                                          subword_mask=batch_subword_mask,
                                          word_mask=batch_word_mask,
                                          word_info=batch_word_info,
                                          x_w2v=batch_w2v_idx,
                                          x_fasttext=batch_ft_idx,
                                          x_pos=batch_pos_idx,
                                          x_ipa=batch_ipa_idx,
                                          x_phono=batch_phono)

    else:
        #raise ValueError("padding for CSE needs to be done here")
        """
        create batch_CSE_embeddings here (encode and then pad the sequence with np.zeros(cse_dim) then convert it into a float tensor)
        """
        max_dim = max([len(t) for t in batch_texts])
        batch_CSE_embeddings = np.zeros((len(batch_texts), max_dim, model.config.embed_dim))
        for tweet_idx in range(len(batch_texts)):
            tokens = batch_texts[tweet_idx]
            sent_emb = model.cse_gen.get_emb(tokens)
            batch_CSE_embeddings[tweet_idx][:sent_emb.shape[0]] = sent_emb
            
        batch_CSE_embeddings = T.tensor(batch_CSE_embeddings).float().to(device)


        ## POTENTIALL SHOULD WORK IF YOU IMPORT from CSETagger in train.py
        if train:
            model = model.train()

            predictions, loss = model(x=batch_CSE_embeddings,
                                      y=batch_labels,
                                      word_mask=batch_word_mask,
                                      x_w2v=batch_w2v_idx,
                                      x_fasttext=batch_ft_idx,
                                      x_pos=batch_pos_idx,
                                      x_ipa=batch_ipa_idx,
                                      x_phono=batch_phono)

        else:
            model = model.eval()

            with T.no_grad():
                predictions, loss = model(x=batch_CSE_embeddings,
                                          y=batch_labels,
                                          word_mask=batch_word_mask,
                                          x_w2v=batch_w2v_idx,
                                          x_fasttext=batch_ft_idx,
                                          x_pos=batch_pos_idx,
                                          x_ipa=batch_ipa_idx,
                                          x_phono=batch_phono)

    
    return predictions, loss
