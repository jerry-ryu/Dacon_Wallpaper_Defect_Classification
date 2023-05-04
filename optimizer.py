import torch

_optimizer_entrypoints = {
    'Adam': torch.optim.Adam
}

def optimizer_entrypoint(optimizer_name):
    return _optimizer_entrypoints[optimizer_name]


def is_optimizer(optimizer_name):
    return optimizer_name in _optimizer_entrypoints


def create_optimizer(optimizer_name, **kwargs):
    if is_optimizer(optimizer_name):
        create_fn = optimizer_entrypoint(optimizer_name)
        optimizer = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % optimizer_name)
    return optimizer