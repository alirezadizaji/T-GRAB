"""
Trainer class
"""

from abc import ABC, abstractmethod
import argparse
import timeit
from typing import Any, Dict, List, Tuple
import os
import shutil
import os.path as osp
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader as torchDataLoader
from tgb.utils.utils import save_results


class Trainer(ABC):
    def __init__(self):
        # Set arguments
        self.args: argparse.Namespace = self._get_args()
        print("INFO: Arguments:", self.args)

        self.device = torch.device("cpu")

        # Data loading
        self.train_loader, self.val_loader, self.test_loader, self.test_inductive_loader = self.create_data_loaders()
        assert isinstance(self.train_loader, torchDataLoader), f"self.train_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.train_loader)} instead."
        assert isinstance(self.val_loader, torchDataLoader), f"self.val_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.val_loader)} instead."
        assert isinstance(self.test_loader, torchDataLoader), f"self.test_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.test_loader)} instead."
        assert isinstance(self.test_inductive_loader, torchDataLoader), f"self.test_inductive_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.test_inductive_loader)} instead."
        self.script_name = os.path.splitext(os.path.basename(__file__))[0]

        # self._clear_results_dirs()

        # Make save directories (if not existed already)
        os.makedirs(self._get_run_save_dir(), exist_ok=True)
        os.makedirs(self._get_results_json_filedir(), exist_ok=True)

    def _clear_results_dirs(self):
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
     
    @property
    def run_dir(self):
        return os.path.join(
            self._get_run_save_dir(),
            self._get_model_card(),
        )

    @abstractmethod
    def create_data_loaders(self) -> Tuple[torchDataLoader, torchDataLoader, torchDataLoader, torchDataLoader]:
        """ This function returns three dataloaders: train/validation/test loaders """
        pass
    
    @property
    def results_path(self) -> str:
        return os.path.join(self.args.root_load_save_dir, 'lab', 'TSA')
    
    @property
    def val_first_metric(self) -> str:
        """ First metric to pick the best model based on """
        return self.args.val_first_metric
    
    @property
    def model_params(self) -> List[str]:
        return self._model_params

    @property
    def choose_best_metric_op(self) -> str:
        """ This property returns the comparison operator that compares 'best validation metric so far' located on the right hand side (rhs) of the opration, 
        with 'current validation metric' on the left hand side (lhs) of operation."""
        return ">"

    def _get_run_save_dir(self) -> str:
        """ This function returns the directory where all results, including json files, model weights, etc are stored.
        Basically, the best hierarchy of running save directory should look as following:
        [Result root dir =`res`]:
            [`DTDG` training results]:
                [Task1: Link prediction training results]:
                    [Memory-node trainer results]:
                        [dataset1 name]:
                            [model1 name = e.g. GCLSTM]
                            [model2 name = EvolveGCN]
                            ...
                        [dataset2 name]:
                            ...
                        ...
                    [Periodicity training results]:
                        ...
                [Task2: Dynamic Graph classification training results]:
                    ...
                [Task3: Dynamic node classification training results]:
                    ...
                [Task4]
                ...
            [`CTDG` training results]:
                ...

        
        Please update this function for your own trainer to make it compatible with this hierarchy 

        """
        return os.path.join(
            self.results_path, 
            "res")
    
    def _get_results_json_filedir(self) -> str:
        return os.path.join(
            self.run_dir,
            'saved_json',
        )
    
    @abstractmethod
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass
    
    @staticmethod
    def _set_running_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('-p', '--node-pos', default="kamada_kawai_layout", required=False, help="How to position nodes during visualization.")
        parser.add_argument('-rlsd', '--root-load-save-dir', default=os.getenv('SCRATCH'), required=True, help="Root directory to load data and store results.")
        parser.add_argument('--val-first-metric', type=str, default="avg_f1", help="First metric to evaluate the validation with. Model weights with best evaluation result during the training is saved.")
        return parser
    
    def _get_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser('*** TSA ***', add_help=False)
        
        parser = self.set_model_args(parser)
        self._model_params = list()
        for action in parser._actions:
            self._model_params.append(action.dest)

        parser = self._set_running_args(parser)

        try:
            args = parser.parse_args()
        except:
            parser.print_help()
            sys.exit(0)
            
        return args

    def _get_model_card(self) -> str:
        return '_'.join([f"{p}={getattr(self.args, p)}" for p in self.model_params])
    
    def add_val_tests_info(self, info: Dict[str, Any]) -> None:
        info[f"val {self.val_first_metric}"] = self.val_perf_list[self.val_first_metric]

        for k, v in self.test_perf.items():
            info[f"test {k}"] = v
        
        for k, v in self.test_inductive_perf.items():
            info[f"test-inductive {k}"] = v

    @abstractmethod
    def eval(self, split_mode: str) -> Dict[str, Any]:
        pass

    def _one_run(self):
        print("---------------------------------------------------------------------")
        start_run = timeit.default_timer()

        self.val_perf_list: Dict[str, List[float]] = dict()
        self.test_perf: Dict[str, float] = dict()
        self.test_inductive_perf: Dict[str, float] = dict()

        # validation
        start_val = timeit.default_timer()
        val_metrics = self.eval(split_mode="val")
        print(f"\tval {self.val_first_metric}: {val_metrics[self.val_first_metric]: .4f}", flush=True)

        for k, v in val_metrics.items():
            if k not in self.val_perf_list:
                self.val_perf_list[k] = list()
            self.val_perf_list[k].append(v)

        end_val = timeit.default_timer()
        print(f"\tValidation: Elapsed time (s): {end_val - start_val: .4f}", flush=True)
        
        # Test
        # Run on test set if validation achieved better results. 
        # Two attributes should be defined: 
        start_test = timeit.default_timer()
        test_metrics = self.eval(split_mode="test")
        test_inductive_metrics = self.eval(split_mode="test_inductive")

        for k, v in test_metrics.items():
            self.test_perf[k] = float(v)
        for k, v in test_inductive_metrics.items():
            self.test_inductive_perf[k] = float(v)

        test_time = timeit.default_timer() - start_test

        # report testing
        print(f"\ttest {self.val_first_metric}: {self.test_perf[self.val_first_metric]: .4f}")
        print(f"\ttest_inductive {self.val_first_metric}: {self.test_inductive_perf[self.val_first_metric]: .4f}")
        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
        
        ### SAVE INFO ###
        info = {'test_time': test_time}
        
        for p in self.model_params:
            info[p] = getattr(self.args, p)
        self.add_val_tests_info(info)

        save_results(info
                    ,os.path.join(self._get_results_json_filedir(), "results.json"))

        print(f"INFO: >>>>> Run elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
        print('-------------------------------------------------------------------------------')


    def run(self):
        start_overall = timeit.default_timer()
        self._one_run()
        print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
        print("==============================================================")
