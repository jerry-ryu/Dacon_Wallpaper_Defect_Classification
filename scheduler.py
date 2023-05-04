import torch

_scheduler_entrypoints = {
   'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau
}
_scheduler_options = {
    'ReduceLROnPlateau': {'mode' : "max", 'factor' : 0.5, 'patience' : 2, 'threshold_mode' :'abs', 'min_lr' : 1e-8, 'verbose' : True }
}

def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoints[scheduler_name]


def is_scheduler(scheduler_name):
    return scheduler_name in _scheduler_entrypoints

def scheduler_option(scheduler_name):
    return _scheduler_options[scheduler_name]

def is_scheduler_option(scheduler_name):
    return scheduler_name in _scheduler_options


def create_scheduler(scheduler_name, **kwargs):
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        if is_scheduler_option:
            scheduler_option(scheduler_name).update(**kwargs)
            scheduler = create_fn(**scheduler_option(scheduler_name))
        else:
            scheduler = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % scheduler_name)
    return scheduler