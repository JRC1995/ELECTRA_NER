import numpy as np
import random
import re
import copy


def batcher(sample_tuples,
            pad_types,
            batch_size=32, bucket_size_factor=5,
            sort_by_idx=0, sort=True, shuffle=True,
            SEED=None):

    # sample_tuples = List of different types of lists of samples
    # Example: [train_data, train_labels]

    # PAD types corresponding to different sequences:
    # Example: [5000, 3] # where 5000 could be the pad id of the train data sequences, and 3 could be the id of the pad -corresponding labels for a sequence labelling task (eg. 'O')
    # .....................(Though the batch masks should be used ideally to nullify any effect of PAD. So it doesn't matter too much, but can be useful in certain cases)

    # bucket_size_factor controls "bucketing". We don't want sequences of too different lengths to be batched together (can cause issues if pads not handled well, or can be inefficient -- short sequences will get over padded need more compute even for sorting)
    # To make sure that similar sized sequences can batched together one can sort first and then batch.
    # But sorting can result in less diverse batch. If there are few sequences with the same sequences they will be always in the same batch. Not ideal if we want more "chaos" in the training (could result in some bias in batch updates).
    # bucketing provides a middle way. We first do sorting, and then create buckets out of the sorted samples. The samples within the buckets are shuffled and batched.
    # bucket size = bucket_size_factor * batch_size
    # You can disable "bucketing" by setting sort=False which results in the "bucketing" turning into just more random shuffling for no reason (unless shuffling too is disabled)

    if SEED is not None:
        random.seed(SEED)

    data1 = sample_tuples[0]
    data_len = len(data1)

    for i in range(data_len):
        # print(len(sorted_sample_tuples[0][i]))
        # print(len(sorted_sample_tuples[1][i]))
        assert len(sample_tuples[0][i]) == len(sample_tuples[-1][i])

    def reorder(samples, idx):
        return [samples[i] for i in idx]

    def reorder_all(sample_tuples, idx):
        return [reorder(samples, idx) for samples in sample_tuples]

    if shuffle:
        random_idx = [i for i in range(data_len)]
        random.shuffle(random_idx)
        shuffled_sample_tuples = reorder_all(sample_tuples, random_idx)
    else:
        shuffled_sample_tuples = sample_tuples

    if sort:
        data1 = shuffled_sample_tuples[sort_by_idx]
        true_seq_lens = [len(sample) for sample in data1]
        sorted_idx = np.flip(np.argsort(true_seq_lens), 0)
        sorted_sample_tuples = reorder_all(shuffled_sample_tuples, sorted_idx)
    else:
        sorted_sample_tuples = sample_tuples

    #print("AFTER SORT AND SHUFFLE")
    for i in range(data_len):
        # print(len(sorted_sample_tuples[0][i]))
        # print(len(sorted_sample_tuples[1][i]))
        assert len(sorted_sample_tuples[0][i]) == len(sorted_sample_tuples[-1][i])

    bucket_size = bucket_size_factor*batch_size

    c = 0
    buckets = []
    while c < data_len:

        start = c
        end = c+bucket_size

        if end > data_len:
            end = data_len

        bucket = [samples[start:end] for samples in sorted_sample_tuples]

        buckets.append(bucket)

        c = end

    if shuffle:
        random.shuffle(buckets)

    def max_len_in_span(samples, start, end):
        if isinstance(samples[0], list):
            return max([len(samples[i]) for i in range(start, end)])
        else:
            return -1

    for bucket in buckets:
        data1 = bucket[0]
        bucket_len = len(data1)

        if shuffle:
            random_idx = [i for i in range(bucket_len)]
            random.shuffle(random_idx)
            bucket = reorder_all(bucket, random_idx)

        i = 0
        while i < bucket_len:
            if i+batch_size > bucket_len:
                incr = bucket_len-i
            else:
                incr = batch_size

            max_lens = [max_len_in_span(samples, i, i+incr) for samples in bucket]

            # print(max_lens)

            batch = [[]]*len(bucket)
            batch_masks = [[]]*len(bucket)

            for j in range(i, i+incr):

                sample_type_id = 0
                for samples, max_len, PAD in zip(bucket, max_lens, pad_types):
                    sample = copy.deepcopy(samples[j])
                    if max_len != -1 and PAD is not None:  # -1 means not list type object. No need of padding

                        sample_len = len(sample)

                        if type(PAD) != type(sample[0]):
                            raise ValueError("INVALID PAD TYPE for Sample Type {}: ".format(sample_type_id) +
                                             "PAD data type mismatch")

                        mask = [1]*sample_len
                        while len(sample) < max_len:
                            sample.append(PAD)
                            mask.append(0)
                    else:
                        mask = []

                    batch[sample_type_id] = batch[sample_type_id] + [sample]
                    batch_masks[sample_type_id] = batch_masks[sample_type_id] + [mask]

                    sample_type_id += 1

            i += incr

            # print(batch[2])

            #batch = [np.asarray(batch_samples) for batch_samples in batch]
            #batch_masks = [np.asarray(batch_mask) for batch_mask in batch_masks]

            yield batch, batch_masks
