from enum import Enum
from pathlib import Path

import sys

from numpy.lib.arraysetops import isin
sys.path.append(str(Path(".").resolve().parent))
sys.path.append(str(Path(".").resolve()))
import glob
import os
from re import T
from typing import List, Optional, Tuple, Union
import numpy as np
from tensorflow.core.util import event_pb2
import tensorflow as tf
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from tensorboard.plugins.hparams.api_pb2 import HParamInfo
from tensorboard.plugins.hparams import api as hp
from tensorboard_utility.src.merger.hparams_chosen import serialize_hparams_chosen_params
from tensorboard_utility.src.merger.hparams_infos import serialize_hparams_domains_infos
from tensorboard_utility.src.merger.simple_summary_parser import DescriptionParser, MetricsAccumulator
import tensorflow as tf
from pathlib import Path


def merger(folder_pattern: str, destination_folder: Optional[str] = None, reference_folder : Union[str,int] = -1, clean: str = "none"):
    """
    Merges all the tensorboard files in the folder_pattern into one file. The folder must be free of use (no tensorboard hook on folders)
    :param folder_pattern: The folder pattern to search for tensorboard files.
    :param clean: If delete, will delete the merged files.
    :return: None
    """

    if not folder_pattern.endswith('/'):
        folder_pattern += '/'
    
    # Get all the tensorboard files
    runs_folders = glob.glob(folder_pattern, recursive=False)
    if destination_folder is None:
        destination_folder = runs_folders[-1]+"_merged/"
    if isinstance(destination_folder,str) and not destination_folder.endswith('/'):
        destination_folder += '/'
        
    original_location = Path(os.getcwd())
    description_parser = DescriptionParser()
    metric_accumulator = MetricsAccumulator()
    parameters_intervals: Optional[dict] = None
    chosen_parameters: Optional[dict] = None
    for run_folder in runs_folders:
        os.chdir(str(original_location.joinpath(run_folder)))
        # Iterate over event files of the run
        for summary_path in glob.glob('**/*.v2',recursive=True):
            summary = tf.data.TFRecordDataset(summary_path)
            for serialized_example in summary:
                event = event_pb2.Event.FromString(serialized_example.numpy())

                for value in event.summary.value:
                    if "_hparams_" in value.tag:
                        data = plugin_data_pb2.HParamsPluginData.FromString(value.metadata.plugin_data.content)
                        if "experiment" in value.tag and parameters_intervals is None and Path(run_folder).name == Path(reference_folder).name:
                            parameters_intervals = serialize_hparams_domains_infos(data)
                        elif "session_start_info" in value.tag  and chosen_parameters is None and Path(run_folder).name == Path(reference_folder).name:
                            chosen_parameters = serialize_hparams_chosen_params(data)
                        else:
                            print(f"WARNING: Unused tag {value.tag}")
                    elif ("Description" in value.tag):
                        t = tf.make_ndarray(value.tensor)
                        in_path = Path(os.getcwd()).joinpath(summary_path)
                        description_parser.add(in_path,destination_folder,str(t))
                        break
                    elif value.tag != "keras":
                        t = tf.make_ndarray(value.tensor)
                        metric_accumulator.add(summary_path,metric_name=value.tag,metric_value=float(t))
            metric_accumulator.flush()
        os.chdir(str(original_location))
                  
    try:
        os.mkdir(destination_folder)
    except FileExistsError:
        pass
    os.makedirs(destination_folder+"train",exist_ok=True)
    os.makedirs(destination_folder+"validation",exist_ok=True)
    if parameters_intervals is None:
        raise Exception("Parameters specifications not found")
    if chosen_parameters is None:
        raise Exception("Chosen parameters not found")

    # Write the summaries
    # Write the metrics
    confidence_intervals = metric_accumulator.compute()
    for folder,data_id in zip(["train","validation"],["tr","val"]):
        for bound in ["high","low","mean"]:
            path_summary = Path(destination_folder+folder).joinpath(bound).resolve()
            file_writer = tf.summary.create_file_writer(str(path_summary),filename_suffix=f"hparams_metrics_{data_id}.v2")
            with file_writer.as_default():
                for metric_name, metric_values in confidence_intervals[data_id][bound].items():
                    metric_values = np.array(metric_values)
                    if len(metric_values.shape) != 2:
                        metric_values = np.expand_dims(metric_values,axis=1)
                    for i,metric in enumerate(metric_values.flatten()):
                        tf.summary.scalar(f"{metric_name}", metric, step=i)
                file_writer.flush()
    ## Write structure of the parameters
    with tf.summary.create_file_writer(destination_folder,filename_suffix="hparams_config.v2").as_default():
        hp.hparams_config(
            hparams=[v for v in parameters_intervals.values()],
            metrics=[
                hp.Metric("final_accuracy_low", display_name='final_accuracy_low'),
                hp.Metric("final_accuracy", display_name='final_accuracy'),
                hp.Metric("final_accuracy_high", display_name='final_accuracy_high')
            ],
        )
    # Write actual parameters chosen
    with tf.summary.create_file_writer(destination_folder,filename_suffix="hparams_chosen.v2").as_default():
        hp.hparams(chosen_parameters)
        tf.summary.scalar(f"final_accuracy_low", float(confidence_intervals["val"]["low"]["final_accuracy"].flatten()[0]), step=0)
        tf.summary.scalar(f"final_accuracy", float(confidence_intervals["val"]["mean"]["final_accuracy"].flatten()[0]), step=0)
        tf.summary.scalar(f"final_accuracy_high", float(confidence_intervals["val"]["high"]["final_accuracy"].flatten()[0]), step=0)
    
    os.chdir(original_location)
    # Delete the individual tensorboard files
    if clean == "delete":
        for tensorboard_file in tensorboard_files:
            os.remove(tensorboard_file)
    elif isinstance(clean,Path):
        for tensorboard_file in tensorboard_files:
            os.rename(tensorboard_file,clean.joinpath(Path(tensorboard_file).name))

