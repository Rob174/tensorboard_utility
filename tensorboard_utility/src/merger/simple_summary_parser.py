from shutil import copy
from pathlib import Path
import os
from scipy.stats import t
from scipy.stats import sem
import numpy as np
from typing import Tuple
class DescriptionParser:
    def __init__(self) -> None:
        self.transfered = True
        self.buffer_text = []
    def add(self,input_file,destination_folder,value):
        input_file = Path(input_file)
        destination_folder = Path(destination_folder)
        if value not in self.buffer_text:
            for folder in ["train/","validation/"]:
                destination_file = str(destination_folder.joinpath(folder))#+input_file.parent.parent.name+"_"+input_file.name
                os.makedirs(destination_file,exist_ok=True)
                copy(str(input_file.resolve()),destination_file)


def make_confidence_interval(samples: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes a confidence interval from a set of samples.
    :param samples: The samples to make the confidence interval from.
    :return: Tuple, The confidence interval.
    """
    confidence_intervals = []
    df = samples.shape[0] - 1
    for i in range(samples.shape[1]):
        if np.isinf(samples[:, i]).any():
            confidence_intervals.append((np.inf,np.inf))
        else:
            confidence_intervals.append(t.interval(confidence,df, loc=np.mean(samples[:, i]), scale=sem(samples[:, i])))
    return np.array([v[0] for v in confidence_intervals]), np.array([v[1] for v in confidence_intervals])
class MetricsAccumulator:
    def __init__(self) -> None:
        self.buffer = {"tr":{},"val":{}}
        self.temporary_buffer = {"tr":{},"val":{}}
        
    def add(self,run_folder,metric_name,metric_value):
        if metric_name == "keras":
            return
        buffer_folder = "tr" if "train" in run_folder else "val"
        if metric_name not in self.temporary_buffer[buffer_folder]:
            self.temporary_buffer[buffer_folder][metric_name] = []
        self.temporary_buffer[buffer_folder][metric_name].append(metric_value)
    def flush(self):
        for buffer_folder in ["tr","val"]:
            for metric_name in self.temporary_buffer[buffer_folder]:
                if metric_name not in self.buffer[buffer_folder]:
                    self.buffer[buffer_folder][metric_name] = []
                self.buffer[buffer_folder][metric_name].append(self.temporary_buffer[buffer_folder][metric_name])
        self.temporary_buffer = {"tr":{},"val":{}}
    def compute(self):
        confidence_intervals = {"tr":{"high":{},"low":{}},"val":{"high":{},"low":{}}}
        for buffer_folder in ["tr","val"]:
            for metric_name,metric_values in self.buffer[buffer_folder].items():
                low_bound,high_bound = make_confidence_interval(np.array(metric_values),0.95)
                confidence_intervals[buffer_folder]["high"][metric_name] = high_bound
                confidence_intervals[buffer_folder]["low"][metric_name] = low_bound
        return confidence_intervals



        

