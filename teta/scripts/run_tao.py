""" evaluate.py

Run example:
evaluate.py --USE_PARALLEL False --METRICS TETA --TRACKERS_TO_EVAL qdtrack

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,  # Raises exception and exits with error
        'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
        'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
        'OUTPUT_TEM_RAW_DATA': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/tao/tao_training'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/tao/tao_training'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val'
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

from teta.config import parse_configs
from teta.datasets import TAO
from teta.eval import Evaluator
from teta.metrics import TETA


def evaluate():
    """Evaluate with TETA."""
    eval_config, dataset_config, metrics_config = parse_configs()
    evaluator = Evaluator(eval_config)
    dataset_list = [TAO(dataset_config)]
    metrics_list = []
    metric = TETA(exhaustive=False)
    if metric.get_name() in metrics_config["METRICS"]:
        metrics_list.append(metric)
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")
    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == "__main__":
    freeze_support()
    evaluate()
