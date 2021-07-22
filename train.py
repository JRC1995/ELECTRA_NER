import numpy as np
import random
from dataLoader.batch import batcher
from transformers import BertTokenizerFast, ElectraTokenizerFast
from configs.WNUT_configs import *
from utils.ml_utils import *
from utils.data_utils import *
from utils.metric_utils import *
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import torch as T
import torch.nn as nn
from models.BigTransformerTagger import BigTransformerTagger
from models.CSETagger import CSETagger
from models.layers.BigTransformers.BERT import BertModel
from models.layers.BigTransformers.ELECTRA import ElectraModel
from models.cse_generator import CSEGenerator
import json
import sys
import re

"""
TRY SAVE BY LOSS IN THE FUTURE
"""
"""
IN FUTURE CHECK IF KEEPING TRUE CASES HARMS OR HELPS BERT
"""

"""
IMPORT MODEL HERE
"""


device = T.device('cuda' if T.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Model Name and stuff')
parser.add_argument('--model', type=str, default="CSE",
                    choices=["ELECTRA",
                             "BERT",
                             "BERT_CRF",
                             "BERT_BiLSTM_CRF",
                             "BERT_w2v_BiLSTM_CRF",
                             "BERT_extra_BiLSTM_CRF",
                             "ELECTRA_CRF",
                             "ELECTRA_BiLSTM_CRF",
                             "ELECTRA_w2v_BiLSTM_CRF",
                             "ELECTRA_extra_BiLSTM_CRF",
                             "ELECTRA_extra_CRF",
                             "ELECTRA_extra",
                             "ELECTRA_w2v_extra_BiLSTM_CRF",
                             "CSE",
                             "CSE_CRF",
                             "CSE_BiLSTM_CRF",
                             "CSE_w2v_BiLSTM_CRF",
                             "CSE_w2v_extra_BiLSTM_CRF",
                             "CSE_extra_BiLSTM_CRF"])

parser.add_argument('--dataset', type=str, default="WNUT_2017")
parser.add_argument('--display_step', type=int, default=30)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--fine_tune_lr', type=float, default=-1)
parser.add_argument('--times', type=int, default=1)
parser.add_argument('--mixed_case_training', type=str, default="no",
                    choices=["yes", "no"])

flags = parser.parse_args()
SEED_base_value = 101

"""
CREATE MAPPINGS HERE
"""

if re.match("^BERT|^ELECTRA", flags.model):
    model_dict = {flags.model: BigTransformerTagger}
elif re.match("^CSE", flags.model):
    model_dict = {flags.model: CSETagger}
else:
    raise ValueError("Invalid model")


config_dict = {flags.model: eval("{0}_config".format(flags.model))}

"""
model_dict = {'BERT': BigTransformerTagger,
              'ELECTRA': BigTransformerTagger,
              'ELECTRA_CRF': BigTransformerTagger,
              "ELECTRA_BiLSTM_CRF": BigTransformerTagger,
              'ELECTRA_w2v_BiLSTM_CRF': BigTransformerTagger,
              "ELECTRA_w2v_extra_BiLSTM_CRF": BigTransformerTagger,
              "ELECTRA_extra_BiLSTM_CRF": BigTransformerTagger,
              "ELECTRA_extra": BigTransformerTagger,
              "ELECTRA_extra_CRF": BigTransformerTagger}

config_dict = {'BERT': BERT_config,
               'ELECTRA': ELECTRA_config,
               'ELECTRA_CRF': ELECTRA_CRF_config,
               "ELECTRA_BiLSTM_CRF": ELECTRA_BiLSTM_CRF_config,
               'ELECTRA_w2v_BiLSTM_CRF': ELECTRA_w2v_BiLSTM_CRF_config,
               'ELECTRA_w2v_extra_BiLSTM_CRF': ELECTRA_w2v_extra_BiLSTM_CRF_config,
               "ELECTRA_extra_BiLSTM_CRF": ELECTRA_extra_BiLSTM_CRF_config,
               "ELECTRA_extra": ELECTRA_extra_config,
               "ELECTRA_extra_CRF": ELECTRA_extra_CRF_config}
"""

config = config_dict[flags.model]
config = config()

if flags.lr >= 0:
    config.lr = flags.lr

if flags.fine_tune_lr >= 0:
    config.fine_tune_lr = flags.fine_tune_lr

display_step = flags.display_step

print('Dataset: {}'.format(flags.dataset))
print("Model Name: {}".format(flags.model))
print("Total Runs: {}".format(flags.times))
print("Learning Rate: {}".format(config.lr))
print("Fine-Tune Learning Rate: {}".format(config.fine_tune_lr))
print("Mixed-Case Training: {}".format(flags.mixed_case_training))
print("Display Step: {}".format(flags.display_step))
print("SEED base value: {}".format(SEED_base_value))


common_data_path = "processed_data/{}/vocab_and_embd.pkl".format(flags.dataset)
if flags.mixed_case_training.lower() == "no":
    train_data_path = "processed_data/{}/train_data.json".format(flags.dataset)
else:
    train_data_path = "processed_data/{}/train_mixed_data.json".format(flags.dataset)
dev_data_path = "processed_data/{}/dev_data.json".format(flags.dataset)
test_data_path = "processed_data/{}/test_data.json".format(flags.dataset)

checkpoint_directory = "saved_params/{}/".format(flags.dataset)
Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)

log_directory = os.path.join("logs", "{}".format(flags.dataset))
Path(log_directory).mkdir(parents=True, exist_ok=True)

keys = ['labels2idx', 'segment_labels2idx',
        'w2v_vocab2idx', 'ft_vocab2idx', 'ipa2idx', 'pos2idx',
        'w2v_embeddings', 'ft_embeddings']

labels2idx, segment_labels2idx,\
    w2v_vocab2idx, ft_vocab2idx, ipa2idx, pos2idx, \
    w2v_embeddings, ft_embeddings = load_data(common_data_path, 'rb', 'pickle', keys=keys)


idx2labels = {v: k for k, v in labels2idx.items()}

"""
DETERMINES WHAT TO LOAD AND IN WHICH ORDER. NEEDS TO MAKE CHANGES IF YOU WANT TO LOAD SOMETHING ELSE
"""
keys = ["sequence",
        "w2v_feats", "fasttext_feats",
        "pos_tags",
        "ipa_feats", "phono_feats",
        "labels", "segment_labels"]

"""
sequence = variable length natural language sequences
w2v_feats = variable length sequences in int format where int id correspond to a word2vec vector (mapped to a word in w2v_vocab2idx)
fasttext_feats = same as above but for fasttext
pos_tags = same as above but int id corresponds to the pos tag of the corresponding word. the id is associated to pos2idx (mapping between id and pos tags). Need to create random embeddings for pos tags.
ipa_feats = character level features will be padded and batched to batch_size x sequence_len x word_len. int format where id correspond to a specific ipa alphabet in ipa2idx mapping. Need to create a randomly initialized embedding.
phono_feats = same as above but each character is represented as a float vector of 22 dimensions instead (can be directly treated as char-level embeddings)
labels = variable length sequence labels for the corresponding sequences. int format. id correspond to a particular label (mapping in labels2idx)
segment_label = we can ignore it for now. Can be later used for multi-tasking for entity-segmentation task (where we do not predict the type of the entity just the boundaries)
"""

"""
For more about load_data see: utils/data_utils.py
"""
train_sample_tuples = load_data(train_data_path, 'r', 'json', keys=keys)
val_sample_tuples = load_data(dev_data_path, 'r', 'json', keys=keys)
test_sample_tuples = load_data(test_data_path, 'r', 'json', keys=keys)

MAX_CHAR_LEN = len(train_sample_tuples[4][0][0])

IPA_PAD = [0]*MAX_CHAR_LEN


PHONO_PAD = [0]*config.phono_feats_dim
PHONO_PAD = [PHONO_PAD]*MAX_CHAR_LEN

if "bert" in flags.model.lower() or "electra" in flags.model.lower():
    if "bert" in flags.model.lower():
        BigModel = BertModel.from_pretrained(config.embedding_path,
                                             output_hidden_states=True,
                                             output_attentions=False)

        tokenizer = BertTokenizerFast.from_pretrained(config.embedding_path,
                                                      output_hidden_states=True,
                                                      output_attentions=False)
    elif "electra" in flags.model.lower():

        BigModel = ElectraModel.from_pretrained(config.embedding_path,
                                                output_hidden_states=True,
                                                output_attentions=False)

        tokenizer = ElectraTokenizerFast.from_pretrained(config.embedding_path,
                                                         output_hidden_states=True,
                                                         output_attentions=False)

    pad_types = [None, w2v_vocab2idx['<pad>'], ft_vocab2idx['<pad>'],
                 pos2idx['G'], IPA_PAD, PHONO_PAD, labels2idx["O"], segment_labels2idx["O"]]

else:
    cse_gen = CSEGenerator(config.use_forward, config.use_backward)
    tokenizer = None
    """
    Probably need to do nothing for CSE here
    text sequences will not be padded (can be padded later after embedding)
    will need to change things if using precomputed embeddings
    """
    pad_types = [None, w2v_vocab2idx['<pad>'], ft_vocab2idx['<pad>'],
                 pos2idx['G'], IPA_PAD, PHONO_PAD, labels2idx["O"], segment_labels2idx["O"]]


def run(time, display_params=False):

    global model_dict
    global flags
    global config
    global device
    global checkpoint_directory, log_directory
    global BigModel
    global w2v_embeddings, ft_embeddings
    global ft_vocab2idx, w2v_vocab2idx, pos2idx, ipa2idx, labels2idx

    mixed_string = "" if flags.mixed_case_training.lower() == "no" else "mixed_case_"

    checkpoint_path = os.path.join(
        checkpoint_directory, "{}_{}run{}.pt".format(flags.model, mixed_string, time))

    log_path = os.path.join(log_directory,
                            "{}_{}run{}.json".format(flags.model, mixed_string, time))

    # print(checkpoint_path)

    # print("Model: {}".format(config.model_name))

    NamedEntitiyRecognizer = model_dict[flags.model]

    """
    May need to make changes here and may be some conditional statements
    """

    if 'bert' in flags.model.lower() or 'electra' in flags.model.lower():

        if config.use_w2v:
            classic_embeddings = w2v_embeddings
            word_pad_id = w2v_vocab2idx['<pad>']
        elif config.use_fasttext:
            classic_embeddings = ft_embeddings
            word_pad_id = ft_vocab2idx['<pad>']
        else:
            classic_embeddings = None
            word_pad_id = None

        if config.use_pos_tags:
            pos_vocab_size = len(pos2idx)
        else:
            pos_vocab_size = None

        if config.use_char_feats:
            ipa_vocab_size = len(ipa2idx)
        else:
            ipa_vocab_size = None

        model = NamedEntitiyRecognizer(BigTransformer=BigModel,
                                       classes_num=len(labels2idx),
                                       config=config,
                                       device=device,
                                       classic_embeddings=classic_embeddings,
                                       word_pad_id=word_pad_id,
                                       pos_vocab_size=pos_vocab_size,
                                       ipa_vocab_size=ipa_vocab_size)

    else:
        """
        Put CSE code here

        """

        if config.use_w2v:
            classic_embeddings = w2v_embeddings
            word_pad_id = w2v_vocab2idx['<pad>']
        elif config.use_fasttext:
            classic_embeddings = ft_embeddings
            word_pad_id = ft_vocab2idx['<pad>']
        else:
            classic_embeddings = None
            word_pad_id = None

        if config.use_pos_tags:
            pos_vocab_size = len(pos2idx)
        else:
            pos_vocab_size = None

        if config.use_char_feats:
            ipa_vocab_size = len(ipa2idx)
        else:
            ipa_vocab_size = None

        model = NamedEntitiyRecognizer(cse_gen,
                                       classes_num=len(labels2idx),
                                       config=config,
                                       device=device,
                                       classic_embeddings=classic_embeddings,
                                       word_pad_id=word_pad_id,
                                       ipa_vocab_size=ipa_vocab_size,
                                       pos_vocab_size=pos_vocab_size)

    model = model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    parameter_count = param_count(parameters)

    print("\n\nParameter Count: {}\n\n".format(parameter_count))
    if display_params:
        param_display_fn(model)

    print("RUN: {}\n\n".format(time))

    run_epochs(model, config, checkpoint_path, log_path)


def run_epochs(model, config, checkpoint_path, log_path):
    """

    raise ValueError(
        "Have you remembered to save the whole epoch log? (both dump output and in a dict)")
    """

    global train_sample_tuples, val_sample_tuples, test_sample_tuples

    train_actual_iters = count_actual_iterations(train_sample_tuples[0], config)
    val_actual_iters = count_actual_iterations(val_sample_tuples[0], config)
    test_actual_iters = count_actual_iterations(test_sample_tuples[0], config)

    train_effective_iters = count_effective_iterations(train_sample_tuples[0], config)
    val_effective_iters = count_effective_iterations(val_sample_tuples[0], config)
    test_effective_iters = count_effective_iterations(test_sample_tuples[0], config)

    # print(train_iters)

    optimizer = load_LRangerMod(model,
                                config=config)  # misleading just running AdamW now

    load = 'n'  # input("\nLoad checkpoint? y/n: ")
    print("\n")
    if load.lower() == 'y':

        print('Loading pre-trained weights for the model...')

        checkpoint = T.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        past_epoch = checkpoint['past epoch']
        train_F1s = checkpoint['train_F1s']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_F1s = checkpoint['val_F1s']
        test_losses = checkpoint['test_losses']
        test_F1s = checkpoint['test_F1s']

        impatience = checkpoint['impatience']

        best_val_loss = min(val_losses)
        best_val_F1 = max(val_F1s)

        print('\nRESTORATION COMPLETE\n')

    else:
        past_epoch = 0

        train_F1s = []
        train_losses = []
        val_losses = []
        val_F1s = []
        test_losses = []
        test_F1s = []

        best_val_loss = math.inf
        best_val_F1 = -math.inf
        impatience = 0

    optimizer.zero_grad()

    # with tqdm(total=config.epochs-past_epoch, desc='Epoch', position=0) as pbar:

    for epoch in range(past_epoch, config.epochs):

        print("TRAINING\n")

        train_loss, train_F1 = run_batches(train_sample_tuples,
                                           epoch=epoch,
                                           model=model,
                                           optimizer=optimizer,
                                           config=config,
                                           generator_len=train_actual_iters,
                                           train=True,
                                           desc='Train Batch')

        train_losses.append(train_loss)
        train_F1s.append(train_F1)

        print("VALIDATING\n")

        val_loss, val_F1 = run_batches(val_sample_tuples,
                                       epoch=epoch,
                                       model=model,
                                       optimizer=optimizer,
                                       config=config,
                                       generator_len=val_actual_iters,
                                       train=False,
                                       desc='Validation Batch')

        val_losses.append(val_loss)
        val_F1s.append(val_F1)

        print("TESTING\n")

        test_loss, test_F1 = run_batches(test_sample_tuples,
                                         epoch=epoch,
                                         model=model,
                                         optimizer=optimizer,
                                         config=config,
                                         generator_len=test_actual_iters,
                                         train=False,
                                         desc='Test Batch')

        test_losses.append(test_loss)
        test_F1s.append(test_F1)

        print("EPOCH SUMMARY\n")

        print("Model: {}, Epoch: {:3d}, ".format(config.model_name, epoch) +
              "Mean Train Loss: {:.3f}, ".format(train_loss) +
              "Mean Train F1: {:.3f}".format(train_F1))

        print("Model: {}, Epoch: {:3d}, ".format(config.model_name, epoch) +
              "Mean Validation Loss: {:.3f}, ".format(val_loss) +
              "Validation F1: {:.3f}".format(val_F1))

        print("Model: {}, Epoch: {:3d}, ".format(config.model_name, epoch) +
              "Mean Test Loss: {:.3f}, ".format(test_loss) +
              "Test F1: {:.3f}".format(test_F1))

        impatience += 1
        save_flag = 0

        if val_loss < best_val_loss:
            impatience = 0
            best_val_loss = val_loss

        if val_F1 >= best_val_F1:
            impatience = 0
            best_val_F1 = val_F1
            save_flag = 1

        print("Impatience Level: {}\n\n".format(impatience))

        if save_flag == 1:

            T.save({
                'past epoch': epoch+1,
                'train_losses': train_losses,
                'train_F1s': train_F1s,
                'val_losses': val_losses,
                'val_F1s': val_F1s,
                'test_losses': test_losses,
                'test_F1s': test_F1s,
                'impatience': impatience,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)

            print("Checkpoint Created!\n\n")

        if impatience >= config.early_stop_patience:
            break

    log_dict = {}
    log_dict["train_losses"] = train_losses
    log_dict["train_F1s"] = train_F1s
    log_dict["val_losses"] = val_losses
    log_dict["val_F1s"] = val_F1s
    log_dict["test_losses"] = test_losses
    log_dict["test_F1s"] = test_F1s

    with open(log_path, "w") as fp:
        json.dump(log_dict, fp)


def run_batches(sample_tuples, epoch,
                model, optimizer, config,
                generator_len,
                train=True, scheduler=None,
                desc=None):

    global display_step
    global pad_types
    global tokenizer
    global idx2labels

    accu_step = config.total_batch_size//config.train_batch_size

    if desc is None:
        desc = 'Batch'

    losses = []
    F1s = []

    total_tp = 0
    total_pred_len = 0
    total_gold_len = 0

    """

    print("TEST RUN")

    data1 = sample_tuples[0]
    data_len = len(data1)

    print("BEFORE SORT AND SHUFFLE")
    for i in range(data_len):
        print(len(sample_tuples[0][i]))
        print(len(sample_tuples[1][i]))
        assert len(sample_tuples[0][i]) == len(sample_tuples[1][i])

    sample_tuples_copy = copy.deepcopy(sample_tuples)

    for batch, batch_masks in batcher(sample_tuples_copy,
                                      pad_types,
                                      config.train_batch_size,
                                      sort_by_idx=1):
        pass

    print("TEST RUN AGAIN")

    print("BEFORE SORT AND SHUFFLE")
    for i in range(data_len):
        print(len(sample_tuples[0][i]))
        print(len(sample_tuples[1][i]))
        assert len(sample_tuples[0][i]) == len(sample_tuples[1][i])

    for batch, batch_masks in batcher(sample_tuples_copy,
                                      pad_types,
                                      config.train_batch_size,
                                      sort_by_idx=1):
        pass
    """

    #copy_tuples = copy.deepcopy(sample_tuples)

    with tqdm(total=generator_len, desc=desc, position=0) as pbar:

        i = 0

        for batch, batch_masks in batcher(sample_tuples,
                                          pad_types,
                                          config.train_batch_size,
                                          sort_by_idx=1):

            # pbar = tqdm(total=generator_len, desc='Batch', position=0)

            batch_texts = batch[0]
            batch_w2v_idx = batch[1]
            batch_ft_idx = batch[2]
            batch_pos_idx = batch[3]
            batch_ipa_idx = batch[4]
            batch_phono = batch[5]
            batch_labels = batch[6]
            batch_segment_labels = batch[7]

            batch_mask = batch_masks[1]

            """
            IMPLEMENT INSIDE utils/ml_utils.py
            """

            predictions, loss = predict_NER(model=model,
                                            tokenizer=tokenizer,
                                            batch_texts=batch_texts,
                                            batch_w2v_idx=batch_w2v_idx,
                                            batch_ft_idx=batch_ft_idx,
                                            batch_pos_idx=batch_pos_idx,
                                            batch_ipa_idx=batch_ipa_idx,
                                            batch_phono=batch_phono,
                                            batch_labels=batch_labels,
                                            batch_segment_labels=batch_segment_labels,
                                            batch_mask=batch_mask,
                                            device=device,
                                            config=config,
                                            train=train)
            losses.append(loss.item())

            if train:

                loss = loss/accu_step
                loss.backward()

                if (i+1) % accu_step == 0:  # Update accumulated gradients

                    T.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                tp, pred_len, gold_len = eval_stats(predictions,
                                                    batch_labels,
                                                    batch_mask,
                                                    idx2labels)

                prec, rec, F1 = compute_F1(tp, pred_len, gold_len)

                F1s.append(F1)

                if i % display_step == 0:

                    pbar.write("Model: {}, Epoch: {:3d}, Iter: {:5d}, ".format(config.model_name, epoch, i) +
                               "Loss: {:.3f}, F1: {:.3f}".format(loss, F1))

            else:
                tp, pred_len, gold_len = eval_stats(predictions,
                                                    batch_labels,
                                                    batch_mask,
                                                    idx2labels)

                total_tp += tp
                total_pred_len += pred_len
                total_gold_len += gold_len

                if i % display_step == 0:

                    pbar.write("Model: {}, Epoch: {:3d}, Iter: {:5d}, ".format(config.model_name, epoch, i) +
                               "Loss: {:.3f}".format(loss))

            i += 1
            pbar.update(1)

    # print("generator_len", generator_len)
    # print("i", i)

    print("\n\n")

    if train:
        F1 = np.mean(F1s)
    else:
        prec, rec, F1 = compute_F1(total_tp, total_pred_len, total_gold_len)

    #del copy_tuples

    return np.mean(losses), F1


if __name__ == '__main__':
    time = 0
    while time < flags.times:

        if time == 0:
            """
            time_str = input("\nStarting time (0,1,2.....times): ")
            try:
                time = int(time_str)
            except:
                time = 0
            """
            time = 0

        SEED = SEED_base_value+time
        T.manual_seed(SEED)
        random.seed(SEED)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        np.random.seed(SEED)

        run(time, display_params=True)
        time += 1
