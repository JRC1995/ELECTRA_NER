from dataLoader.batch import batcher


type1_samples = [[10, 11, 12], [10], [10, 11, 12, 13], [10, 11]]
type2_samples = [[20, 21, 22], [20], [20, 21, 22, 23], [20, 21]]
type3_samples = [3, 1, 4, 2]

sample_tuples = [type1_samples, type2_samples, type3_samples]
pad_types = [0, -1, None]

i = 0
for batch, batch_masks in batcher(sample_tuples, pad_types, batch_size=2):
    type1_batch = batch[0]
    type2_batch = batch[1]
    type3_batch = batch[2]

    type1_mask = batch_masks[0]
    type2_mask = batch_masks[0]

    print("type1", type1_batch)
    print("type2", type2_batch)
    print("type3", type3_batch)

    i += 1


type1_samples = [[10, 11, 12], [10], [10, 11, 12, 13], [10, 11]]
type2_samples = [[20, 21, 22], [20], [20, 21, 22, 23], [20, 21]]
type3_samples = [3, 1, 4, 2]

sample_tuples = [type1_samples, type2_samples, type3_samples]
pad_types = [None, 0.89, 2]  # Check robustness to invalid Pad values. Should RAISE errors

i = 0
for batch, batch_masks in batcher(sample_tuples, pad_types, batch_size=2):
    type1_batch = batch[0]
    type2_batch = batch[1]
    type3_batch = batch[2]

    type1_mask = batch_masks[0]
    type2_mask = batch_masks[0]

    print("type1", type1_batch)
    print("type2", type2_batch)
    print("type3", type3_batch)

    i += 1
