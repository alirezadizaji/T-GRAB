"""
Trainer class
"""

from abc import ABC, abstractmethod
import argparse
import json
import gc
import timeit
from typing import Any, Dict, List, Tuple
import os
import os.path as osp
import shutil
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader as torchDataLoader
from tgb.utils.utils import set_random_seed, save_results
import wandb

from ..utils import  EarlyStopMonitor, init_summary_writer


class Trainer(ABC):
    def __init__(self):
        # Set arguments
        self.args: argparse.Namespace = self._get_args()
        print("INFO: Arguments:", self.args)

        # Set device
        if self.args.device == "default":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)

        # Data loading
        self.train_loader, self.val_loader, self.test_loader, self.test_inductive_loader = self.create_data_loaders()
        assert isinstance(self.train_loader, torchDataLoader), f"self.train_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.train_loader)} instead."
        assert isinstance(self.val_loader, torchDataLoader), f"self.val_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.val_loader)} instead."
        assert isinstance(self.test_loader, torchDataLoader), f"self.test_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.test_loader)} instead."
        assert isinstance(self.test_inductive_loader, torchDataLoader), f"self.test_inductive_loader should be an instance of `torch.utils.data.DataLoader`; got {type(self.test_inductive_loader)} instead."
        
        self.script_name = os.path.splitext(os.path.basename(__file__))[0]

        self.epoch: int = None

        self.run_idx = 0

        # DO NOT clear results when is in evaluation mode
        if self.args.clear_results and not self.args.eval_mode:
            self._clear_results_dirs()

        # Make save directories (if not existed already)
        os.makedirs(self._get_run_save_dir(), exist_ok=True)
        os.makedirs(self._get_results_json_filedir(), exist_ok=True)
        os.makedirs(self._get_save_model_dir(), exist_ok=True)
        os.makedirs(self.tx_summary_path, exist_ok=True)

        wandb_name = self.run_dir.replace(self.results_path, "")

        try:
            print(f"Initializing Wandb run...")
            wandb.init(project=self.args.wandb_project, 
                    entity=self.args.wandb_entity, 
                    name=wandb_name, 
                    dir=self.results_path,
                    config={"log_interval": self.args.wandb_log_interval},
                    settings=wandb.Settings(console="off"))
            print("Wandb run initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Wandb run: {e}")
            print("Continuing without Wandb logging...")
            raise Exception(f"Failed to initialize Wandb run: {e}")

    @property
    def run_dir(self):
        return os.path.join(
            self._get_run_save_dir(),
            self._get_model_card(),
            self._run_id(),
        )

    def _clear_results_dirs(self):
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
    
    @abstractmethod
    def create_data_loaders(self) -> Tuple[torchDataLoader, torchDataLoader, torchDataLoader]:
        """ This function returns three dataloaders: train/validation/test loaders """
        pass
    
    @property
    def results_path(self) -> str:
        return os.path.join(self.args.root_load_save_dir, 'lab', 'TSA')
    
    @property
    def tx_summary_path(self) -> str:
        "TensorboardX summary writer path"
        d = os.path.join(self.run_dir,
            "tensorboard",
        )

        os.makedirs(d, exist_ok=True)
        return d

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
    
    def _get_save_model_dir(self) -> str:
        return os.path.join(
            self.run_dir,
            'saved_models',
        )
    
    @abstractmethod
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass
    
    @staticmethod
    def _set_running_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
        parser.add_argument('--num-epoch', type=int, help='Number of epochs', default=100)
        parser.add_argument('--seed', type=int, help='Random seed', default=1)
        parser.add_argument('--patience', type=float, help='Early stopper patience', default=15)
        parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
        parser.add_argument('--num-run', type=int, help='Number of iteration runs', default=1)
        parser.add_argument('-r', '--clear-results', action='store_true')
        parser.add_argument('-wp', '--wandb-project', type=str, default="TSA", help="Wandb project name.")
        parser.add_argument('-we', '--wandb-entity', type=str, default="alirezadizaji24-universit-de-montr-al", help="Wandb entity name.")
        parser.add_argument('-wli', '--wandb-log-interval', type=int, default=30 , help="Wandb log interval in steps.")
        parser.add_argument('-p', '--node-pos', default="kamada_kawai_layout", required=False, help="How to position nodes during visualization.")
        parser.add_argument('-e', '--eval-mode', action="store_true", help="If given, then the model checkpoint is being loaded and only inference is done.")
        parser.add_argument('-rlsd', '--root-load-save-dir', required=True, default=os.getenv('SCRATCH'), help="The root directory to store results and load the data.")
        parser.add_argument('--device', type=str, default="default")
        parser.add_argument('--val-first-metric', type=str, default="avg_f1", help="First metric to evaluate the validation with. Model weights with best evaluation result during the training is saved.")
        parser.add_argument('--train-eval-gap', type=int, default=1, help="Number of epochs between training and evaluation.")
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
    
    def _run_id(self) -> str:
        return f'seed{self.args.seed}_runidx{self.run_idx}'

    @abstractmethod
    def train_for_one_epoch(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def eval_for_one_epoch(self, split_mode: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def early_stopping_checker(self, early_stopper) -> bool:
        pass

    def add_val_tests_info(self, info: Dict[str, Any]) -> None:
        info[f"val {self.val_first_metric}"] = self.val_perf_list[self.val_first_metric]

        for k, v in self.test_perf.items():
            info[f"test {k}"] = v
        
        for k, v in self.test_inductive_perf.items():
            info[f"test-inductive {k}"] = v

    def _one_run(self):
        print("---------------------------------------------------------------------")
        print(f"INFO: >>>>> Run: {self.run_idx} <<<<<")
        start_run = timeit.default_timer()

        # define an early stopper
        save_model_dir = self._get_save_model_dir()
        save_model_card = self._get_model_card()
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_card, 
                                        tolerance=self.args.tolerance, patience=self.args.patience)
        json_save_dir = os.path.join(self._get_results_json_filedir(), "results.json")
        
        # Load the model and resume the training.
        loaded = early_stopper.load_checkpoint(self.model, self.device)
        if loaded:
            with open(json_save_dir, 'r') as json_file:
                file_data = json.load(json_file)
                # convert file_data to list if not
                if type(file_data) is dict:
                    file_data = [file_data]
                for d in file_data[::-1]:
                    # Load the last trained epoch
                    if "epoch" in d:
                        epoch_start = d["epoch"]
                        break
                    # If last epoch not recorded, load the best epoch
                    elif "best_epoch" in d:
                        epoch_start = d["best_epoch"]
                        break
        else:
            epoch_start = 1

        # ==================================================== Train & Validation
        # Records the training performance at different epochs
        self.train_perf_list: Dict[str, List[float]] = dict()

        best_val_first_metric = None
        # Records the validation performance at different epochs
        self.val_perf_list: Dict[str, List[float]] = dict()
        self.test_perf: Dict[str, float] = dict()
        self.test_inductive_perf: Dict[str, float] = dict()

        train_times_l, val_times_l = [], []
        free_mem_l, total_mem_l, used_mem_l = [], [], []
        start_train_val = timeit.default_timer()

        for self.epoch in range(epoch_start, self.args.num_epoch + 1):
            # training
            start_epoch_train = timeit.default_timer()
            
            for model in self.model.values():
                model.train()
            train_metrics = self.train_for_one_epoch()
            
            for k, v in train_metrics.items():
                # Train wandb plot
                wandb.log({"epoch": self.epoch, f"Train-{k}": v})

                if k not in self.train_perf_list:
                    self.train_perf_list[k] = list()
                self.train_perf_list[k].append(v)
                self.writer.add_scalar(f"Train-{k}", v, global_step=self.epoch)

            end_epoch_train = timeit.default_timer()
            print(f"\tTraining elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}", flush=True)
            # checking GPU memory usage
            free_mem, used_mem, total_mem = 0, 0, 0
            if torch.cuda.is_available():
                print("DEBUG: device: {}".format(torch.cuda.get_device_name(0)))
                free_mem, total_mem = torch.cuda.mem_get_info()
                used_mem = total_mem - free_mem
                print("------------Epoch {}: GPU memory usage-----------".format(self.epoch))
                print("Free memory: {}".format(free_mem))
                print("Total available memory: {}".format(total_mem))
                print("Used memory: {}".format(used_mem))
                print("--------------------------------------------")
            
            train_times_l.append(end_epoch_train - start_epoch_train)
            free_mem_l.append(float((free_mem*1.0)/2**30))  # in GB
            used_mem_l.append(float((used_mem*1.0)/2**30))  # in GB
            total_mem_l.append(float((total_mem*1.0)/2**30))  # in GB

            gc.collect()
            torch.cuda.empty_cache()

            if self.epoch % self.args.train_eval_gap == 0:
                for model in self.model.values():
                    model.eval()

                with torch.no_grad():
                    # validation
                    start_val = timeit.default_timer()
                    val_metrics = self.eval_for_one_epoch(split_mode="val")
                    print(f"\tval {self.val_first_metric}: {val_metrics[self.val_first_metric]: .4f}", flush=True)

                    for k, v in val_metrics.items():
                        # Validation wandb plot
                        wandb.log({"epoch": self.epoch, f"Val-{k}": v})

                        if k not in self.val_perf_list:
                            self.val_perf_list[k] = list()
                        self.val_perf_list[k].append(v)

                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f"Validation-{k}", v, global_step=self.epoch)

                    end_val = timeit.default_timer()
                    print(f"\tValidation: Elapsed time (s): {end_val - start_val: .4f}", flush=True)
                    val_times_l.append(end_val - start_val)
                    
                    # Test
                    lhs, rhs = val_metrics[self.val_first_metric], best_val_first_metric
                    eq = " ".join([str(lhs), self.choose_best_metric_op, str(rhs)])
            
                    # Run on test set if validation achieved better results. 
                    # Two attributes should be defined: 
                    if rhs is None or eval(eq):
                        best_val_first_metric = lhs
                        start_test = timeit.default_timer()
                        test_metrics = self.eval_for_one_epoch(split_mode="test")
                        test_inductive_metrics = self.eval_for_one_epoch(split_mode="test_inductive")
                
                        for k, v in test_metrics.items():
                            # Test wandb plot
                            wandb.log({"epoch": self.epoch, f"Test-{k}": v})
                            self.test_perf[k] = float(v)
                        
                        for k, v in test_inductive_metrics.items():
                            # Test inductive wandb plot
                            wandb.log({"epoch": self.epoch, f"Test-inductive-{k}": v})
                            self.test_inductive_perf[k] = float(v)

                        test_time = timeit.default_timer() - start_test

                        early_stopper.save_checkpoint(self.model)

                        train_val_time = timeit.default_timer() - start_train_val
                        print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

                        # report testing
                        print(f"\ttest {self.val_first_metric}: {self.test_perf[self.val_first_metric]: .4f}")
                        print(f"\ttest inductive {self.val_first_metric}: {self.test_inductive_perf[self.val_first_metric]: .4f}")
                        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
                        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
                        
                        ### SAVE INFO ###
                        info = {'run': self.run_idx,
                                'epoch': self.epoch,
                                'seed': self.args.seed,
                                'train_times': train_times_l,
                                'free_mem': free_mem_l,
                                'total_mem': total_mem_l,
                                'used_mem': used_mem_l,
                                'max_used_mem': max(used_mem_l),
                                'val_times': val_times_l,
                                'test_time': test_time,
                                'eval_mode': self.args.eval_mode,
                                'train_val_total_time': np.sum(np.array(train_times_l)) + np.sum(np.array(val_times_l))}

                        for k, v in self.train_perf_list.items():
                            info[f'train_{k}'] = v
                        
                        for p in self.model_params:
                            info[p] = getattr(self.args, p)
                        self.add_val_tests_info(info)

                        save_results(info, json_save_dir)
                        print(f"JSON file saved at {json_save_dir}.", flush=True)
                
                # check for early stopping
                if self.early_stopping_checker(early_stopper):
                    break
        
        wandb.finish()
        print(f"INFO: >>>>> Run: {self.run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
        print('-------------------------------------------------------------------------------')


    def _one_run_inference(self):
        print('-------------------------------------------------------------------------------')
        print(f"INFO: >>>>> Run: {self.run_idx} <<<<<")
        start_run = timeit.default_timer()

        # define an early stopper
        save_model_dir = self._get_save_model_dir()
        save_model_id = self._get_model_card()
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                        tolerance=self.args.tolerance, patience=self.args.patience)

        # Load the model
        early_stopper.load_checkpoint(self.model, self.device)

        for model in self.model.values():
            model.eval()

        with torch.no_grad():
            start_test = timeit.default_timer()
            self.eval_for_one_epoch(split_mode="train")
            self.eval_for_one_epoch(split_mode="val")
            test_metrics = self.eval_for_one_epoch(split_mode="test")
            test_inductive_metrics = self.eval_for_one_epoch(split_mode="test_inductive")
            test_time = timeit.default_timer() - start_test
        
        ## Save INFO
        info = {'run': self.run_idx,
                'seed': self.args.seed,
                'test_time': test_time,
                'eval_mode': self.args.eval_mode}
        for p in self.model_params:
            info[p] = getattr(self.args, p)
        for k, v in test_metrics.items():
            info[f"test {k}"] = float(v)
        for k, v in test_inductive_metrics.items():
            info[f"test-inductive {k}"] = float(v)
        
        json_save_dir = os.path.join(self._get_results_json_filedir(), "results.json")
        save_results(info, json_save_dir)
        print(f"JSON file saved at {json_save_dir}.", flush=True)
        print(f"\tTest performance: {test_metrics}", flush=True)
        print(f"\n\tTest-inductive performance: {test_inductive_metrics}", flush=True)

        # report testing
        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
        
        print(f"INFO: >>>>> Run: {self.run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
        print('-------------------------------------------------------------------------------')

    def run(self):
        start_overall = timeit.default_timer()

        for self.run_idx in range(self.args.num_run):
            # set the seed
            torch.manual_seed(self.run_idx + self.args.seed)
            set_random_seed(self.run_idx + self.args.seed)
        
            # Set summary writer
            self.writer = init_summary_writer(self.tx_summary_path)

            if not self.args.eval_mode:
                self._one_run()
            else:
                self._one_run_inference()

        print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
        print("==============================================================")
