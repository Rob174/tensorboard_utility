from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData

def serialize_hparams_chosen_params(data: HParamsPluginData) -> dict:
    """
    Serializes the hparams chosen params.
    :param data: The hparams plugin data.
    :return: The serialized hparams chosen params.
    """
    try:
        hparams = list(data.session_start_info.hparams.items()) 
    except Exception:
        return {}
    dico = {}
    actions = {
        "bool_value":lambda x: x.bool_value,
        "number_value":lambda x: x.number_value,
        "string_value":lambda x: x.string_value,
        "list_value":lambda x: x.list_value
    }
    for name,value in hparams:
        for field_name,access_fn in actions.items():
            if value.HasField(field_name):
                dico[name] = access_fn(value)
                break
    return dico