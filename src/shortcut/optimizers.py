from torch.optim import Adam, AdamW, SGD

name_to_optimizer = {
    'Adam': Adam,
    'AdamW': AdamW,
    'SGD': SGD,
}


def get_optimizer(name):
    return name_to_optimizer[name]
