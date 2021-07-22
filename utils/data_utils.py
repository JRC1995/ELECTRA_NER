import json
import pickle


def load_data(path, mode, data_lib, keys):

    data_lib = data_lib.lower()

    data_lib = {'pickle': pickle,
                'json': json}.get(data_lib, json)

    with open(path, mode) as fp:
        data = data_lib.load(fp)
        sample_tuples = []
        for key in keys:
            sample_tuples.append(data[key])
    return sample_tuples


def param_count(parameters):
    return sum(p.numel() for p in parameters if p.requires_grad)


def param_display_fn(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.size())
    print("\n\n")


def count_actual_iterations(data, config):
    data_len = len(data)
    iters = data_len//config.train_batch_size
    if data_len % config.train_batch_size > 0:
        iters += 1
    return iters


def count_effective_iterations(data, config):
    data_len = len(data)
    iters = data_len//config.total_batch_size
    if data_len % config.total_batch_size > 0:
        iters += 1
    return iters
