import numpy as np
import random
from dataLoader.batch import batcher
from transformers import BertTokenizerFast, ElectraTokenizerFast
from configs.WNUT_MRC_configs import *
from utils.ml_utils import *
from utils.mrc_ml_utils import *
from utils.data_utils import *
from utils.metric_utils import *
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import torch as T
import torch.nn as nn
from models.BigTransformerMRC import BigTransformerMRC
from models.layers.BigTransformers.BERT import BertModel
from models.layers.BigTransformers.ELECTRA import ElectraModel
import json

"""
TRY SAVE BY LOSS IN THE FUTURE
"""
"""
IN FUTURE CHECK IF KEEPING TRUE CASES HARMS OR HELPS BERT
"""

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Model Name and stuff')
parser.add_argument('--model', type=str, default="ELECTRA_CRF_MRC",
                    choices=["ELECTRA_MRC",
                             "ELECTRA_SL_MRC",
                             "ELECTRA_DSC_MRC",
                             "ELECTRA_CRF_MRC",
                             "ELECTRA_BiLSTM_natural_query",
                             "ELECTRA_BiLSTM_SL_natural_query",
                             "ELECTRA_BiLSTM_CRF_natural_query",
                             "ELECTRA_BiLSTM_MRC",
                             "ELECTRA_BiLSTM_SL_MRC",
                             "ELECTRA_BiLSTM_CRF_MRC"])
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
AUTOMATICALLY SELECT CONFIG BASED ON ARGPARSE COMMAND
"""
config_dict = {flags.model: eval("{0}_config".format(flags.model))}
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


common_data_path = os.path.join("processed_data", "{}".format(
    flags.dataset), "vocab_and_embd_MRC.pkl")
if flags.mixed_case_training.lower() == "no":
    train_data_path = "processed_data/{}/train_data_MRC.json".format(flags.dataset)
else:
    train_data_path = "processed_data/{}/train_mixed_data_MRC.json".format(flags.dataset)
dev_data_path = "processed_data/{}/dev_data_MRC.json".format(flags.dataset)
test_data_path = "processed_data/{}/test_data_MRC.json".format(flags.dataset)

checkpoint_directory = os.path.join("saved_params", "{}_MRC".format(flags.dataset))
Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)

Path("output/").mkdir(parents=True, exist_ok=True)

log_directory = os.path.join("logs", "{}_MRC".format(flags.dataset))
Path(log_directory).mkdir(parents=True, exist_ok=True)


"""
DETERMINES WHAT TO LOAD AND IN WHICH ORDER. NEEDS TO MAKE CHANGES IF YOU WANT TO LOAD SOMETHING ELSE
"""

keys = ['segment_labels2idx', 'types2idx', 'type2natural_query']

segment_labels2idx,\
    types2idx, type2natural_query = load_data(common_data_path, "rb", "pickle", keys=keys)


idx2segment_labels = {v: k for k, v in segment_labels2idx.items()}

keys = ["sequence",
        "queries",
        "natural_queries",
        'span_start_binary',
        'span_end_binary',
        'segment_labels']

"""
sequence = variable length natural language sequences
queries = each sample is an int id representing a specific query (mapped to a type through types2idx)
natural_queries = variable length natural language sequence queries (representing entity types --- mapping in type2natural_query)
span_start_binary = like a sequence label where labels are 1 if the token in position is the start of an answer (entity) span otherwise 0
span_end_binary = like a sequence label where labels are 1 if the token in position is the end of an answer (entity) span otherwise 0
segment_label = in BIO format instead of start indices or end indices
"""

"""
For more about load_data see: utils/data_utils.py
"""
train_sample_tuples = load_data(train_data_path, 'r', 'json', keys=keys)
val_sample_tuples = load_data(dev_data_path, 'r', 'json', keys=keys)
test_sample_tuples = load_data(test_data_path, 'r', 'json', keys=keys)


BigModel = ElectraModel.from_pretrained(config.embedding_path,
                                        output_hidden_states=True,
                                        output_attentions=False)

tokenizer = ElectraTokenizerFast.from_pretrained(config.embedding_path,
                                                 output_hidden_states=True,
                                                 output_attentions=False)

pad_types = [None, None, None, 0, 0, segment_labels2idx["O"]]


def run(time, display_params=False):

    global flags
    global config
    global device
    global checkpoint_directory, log_directory
    global BigModel
    global segment_labels2idx, types2idx

    mixed_string = "" if flags.mixed_case_training.lower() == "no" else "mixed_case_"

    checkpoint_path = os.path.join(checkpoint_directory,
                                   "{}_{}run{}.pt".format(flags.model, mixed_string, time))

    log_path = os.path.join(log_directory,
                            "{}_{}run{}.json".format(flags.model, mixed_string, time))

    # print(checkpoint_path)

    # print("Model: {}".format(config.model_name))
    model = BigTransformerMRC(BigTransformer=BigModel,
                              segment_labels2idx=segment_labels2idx,
                              config=config,
                              device=device,
                              query_vocab_size=len(types2idx))

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
    global log_directory
    global flags

    train_actual_iters = count_actual_iterations(train_sample_tuples[0], config)
    # count_actual_iterations(val_sample_tuples[0], config)
    val_actual_iters = len(val_sample_tuples[0])//config.val_batch_size
    # count_actual_iterations(test_sample_tuples[0], config)
    test_actual_iters = len(test_sample_tuples[0])//config.val_batch_size

    train_effective_iters = count_effective_iterations(train_sample_tuples[0], config)
    val_effective_iters = count_effective_iterations(val_sample_tuples[0], config)
    test_effective_iters = count_effective_iterations(test_sample_tuples[0], config)

    # print(train_iters)

    optimizer = load_LRangerMod(model,
                                config=config)  # misleading just running AdamW now

    print('Loading pre-trained weights for the model...')

    checkpoint = T.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('\nRESTORATION COMPLETE\n')

    # with tqdm(total=config.epochs-past_epoch, desc='Epoch', position=0) as pbar:

    print("TESTING\n")

    test_loss, test_F1 = run_batches(test_sample_tuples,
                                     epoch=0,
                                     model=model,
                                     optimizer=optimizer,
                                     config=config,
                                     generator_len=test_actual_iters,
                                     train=False,
                                     desc='Test Batch')


def run_batches(sample_tuples, epoch,
                model, optimizer, config,
                generator_len,
                train=True, scheduler=None,
                desc=None):

    global display_step
    global pad_types
    global tokenizer
    global idx2segment_labels

    accu_step = config.total_batch_size//config.train_batch_size

    if train:
        batch_size = config.train_batch_size
    else:
        batch_size = config.val_batch_size

    if desc is None:
        desc = 'Batch'

    losses = []
    F1s = []

    total_tp = 0
    total_pred_len = 0
    total_gold_len = 0

    f = open("output/out_{}.txt".format(flags.model), "w")
    f.write('')
    f.close()

    with tqdm(total=generator_len, desc=desc, position=0) as pbar:

        i = 0

        for batch, batch_masks in batcher(sample_tuples,
                                          pad_types,
                                          batch_size,
                                          sort_by_idx=3):

            # pbar = tqdm(total=generator_len, desc='Batch', position=0)

            batch_texts = batch[0]
            batch_queries_idx = batch[1]
            batch_natural_queries = batch[2]
            batch_labels_start = batch[3]
            batch_labels_end = batch[4]
            batch_segment_labels = batch[5]
            batch_mask = batch_masks[3]

            """
            print(batch_texts[0])
            print(batch_queries_idx[0])
            print(batch_natural_queries[0])
            print(batch_labels_start[0])
            print(batch_labels_end[0])
            print(batch_segment_labels[0])
            print(batch_mask[0])
            """

            """
            IMPLEMENT INSIDE utils/mrc_ml_utils.py
            """

            predictions, loss = predict_NER_MRC(model=model,
                                                tokenizer=tokenizer,
                                                batch_texts=batch_texts,
                                                batch_queries_idx=batch_queries_idx,
                                                batch_natural_queries=batch_natural_queries,
                                                batch_labels_start=batch_labels_start,
                                                batch_labels_end=batch_labels_end,
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
                                                    batch_segment_labels,
                                                    batch_mask,
                                                    idx2segment_labels)

                prec, rec, F1 = compute_F1(tp, pred_len, gold_len)

                F1s.append(F1)

                if i % display_step == 0:

                    pbar.write("Model: {}, Epoch: {:3d}, Iter: {:5d}, ".format(config.model_name, epoch, i) +
                               "Loss: {:.3f}, F1: {:.3f}".format(loss, F1))

            else:

                f = open("output/out_{}.txt".format(flags.model), "a")
                for prediction_sample, gold_sample, mask in zip(predictions, batch_segment_labels, batch_mask):
                    true_seq_len = sum(mask)
                    prediction_sample = prediction_sample[0:true_seq_len]
                    gold_sample = gold_sample[0:true_seq_len]
                    for pred, gold in zip(prediction_sample, gold_sample):
                        f.write(
                            "test NNP "+str(idx2segment_labels[gold])+" "+str(idx2segment_labels[pred])+"\n")
                f.close()

                tp, pred_len, gold_len = eval_stats(predictions,
                                                    batch_segment_labels,
                                                    batch_mask,
                                                    idx2segment_labels)

                total_tp += tp
                total_pred_len += pred_len
                total_gold_len += gold_len

                if i % display_step == 0:

                    pbar.write("Model: {}, Epoch: {:3d}, Iter: {:5d}, ".format(config.model_name, epoch, i) +
                               "Loss: {:.3f}".format(loss))

            i += 1
            pbar.update(1)

    print("\n\n")

    if train:
        F1 = np.mean(F1s)
    else:
        prec, rec, F1 = compute_F1(total_tp, total_pred_len, total_gold_len)

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
