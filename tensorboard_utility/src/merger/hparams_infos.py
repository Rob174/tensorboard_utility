from typing import Dict
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from tensorboard.plugins.hparams.api_pb2 import HParamInfo
from tensorboard.plugins.hparams import api as hp
from enum import Enum

class TypeEnum(int,Enum):
    BOOL = 2
    STRING = 1
    FLOAT = 3
def serialize_real_interval(infos: HParamInfo) -> hp.HParam:
    try:
        min = infos.domain_discrete.min_value
    except AttributeError:
        min = 0.
    try:
        max = infos.domain_discrete.max_value
    except AttributeError:
        max = 0.
    return hp.HParam(infos.name, hp.RealInterval(min,max))

def serialize_stringBool_choice(infos: HParamInfo) -> hp.HParam:
    return hp.HParam(infos.name, hp.Discrete(list(infos.domain_discrete.items())))
def serialize_hparams_domains_infos(data: HParamsPluginData) -> Dict[str, hp.HParam]:
    """
    Serializes the hparams domains infos.
    :param data: The hparams plugin data.
    :return: The serialized hparams domains infos.
    """
    try:
        hparams = [data.experiment.hparam_infos[i] for i in range(len(data.experiment.hparam_infos))]
    except Exception:
        return {}
    dico_parameters = {}

    for hparam in hparams:
        if hparam.HasField("domain_interval") and hparam.type == TypeEnum.FLOAT:
            dico_parameters[hparam.name] = serialize_real_interval(hparam)
        elif hparam.HasField("domain_discrete") and (hparam.type == TypeEnum.STRING or hparam.type == TypeEnum.BOOL):
            dico_parameters[hparam.name] = serialize_stringBool_choice(hparam)
        else:
            print(f"WARNING: Unsupported hparam {hparam}")
    return dico_parameters