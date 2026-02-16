from shortcut.models.encoder import (EfficientNetB1, ResNetEncoder,
                                     SimpleEncoder)
from shortcut.models.subspace_disentanglement import (MINE,
                                                      AdversarialClassifierGRL,
                                                      Baseline, dCor, MMD)


def get_method(cfg: dict):
    methods = {
        "Baseline": Baseline,
        "Rebalancing": Baseline,
        "MINE": MINE,
        "dCor": dCor,
        "adv_cl": AdversarialClassifierGRL,
        "MMD": MMD,
    }
    return methods[cfg.model.method](cfg)


def get_encoder(cfg: dict):
    encoders = {
        "resnet": ResNetEncoder,
        "efficientnet_b1": EfficientNetB1,
        "simple_encoder": SimpleEncoder,
    }
    return encoders[cfg.model.encoder](cfg)


def get_model(cfg):
    method = get_method(cfg)
    encoder = get_encoder(cfg)
    return method, encoder
