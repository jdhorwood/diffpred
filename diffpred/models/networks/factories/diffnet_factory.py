from diffpred.models.protocols import ArchitectureConfig, DiffNetProtocol
from diffpred.models.networks import BaselineDiffusionNet


def diffusion_network_factory(
    conf: ArchitectureConfig, classifier: bool = False
) -> DiffNetProtocol:
    if conf.arch_type == "baseline":
        return BaselineDiffusionNet(conf, classifier=classifier)
    else:
        raise NotImplementedError
