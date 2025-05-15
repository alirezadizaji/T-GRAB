from abc import abstractmethod
from argparse import ArgumentParser
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as torchDataLoader
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator


from ....dataset.DTDG.torch_dataset.link_pred.node_feat_static import LinkPredNodeFeatureStaticDataset
from ....model.edgebank_predictor import EdgeBankPredictor
from ....utils import visualizer, NodeFeatType
from ...CTDG.edgebank import EdgeBankTrainer

class LinkPredEdgeBankTrainer(EdgeBankTrainer):
    def __init__(self):
        super(LinkPredEdgeBankTrainer, self).__init__()
        assert 'edgebank' in self.model.keys(), 'self.model should contain a graph learning model, passed as `gnn` keyword.'
        assert isinstance(self.model['edgebank'], EdgeBankPredictor), 'gnn model should be a child class of GraphModel.'

        # Check the dataset name follows the regex pattern
        dataset_regex_pattern, dataset_name_part_to_check = self.get_dataset_regex_pattern()
        import re
        assert re.match(dataset_regex_pattern, dataset_name_part_to_check), f"Given dataset name ({self.args.data}) doesn't match with the dataset regex pattern ({dataset_regex_pattern})."

        # Set MRR evaluator
        self.evaluator = Evaluator(name=self.args.data)
       
        # initiate negative sampler
        self.neg_sampler = NegativeEdgeSampler(dataset_name=self.args.data)

        # Visualizer to output graph generation by link prediction
        self.vis = visualizer(save_dir=self._get_visualization_dir(), num_nodes=self.train_loader.dataset.num_nodes, node_pos=self.args.node_pos)

    @abstractmethod
    def get_dataset_regex_pattern(self):
        """ Each task in link prediction supports a group of datasets that follow a regex pattern. To understand current supported tasks,
        please checkout subfolders of `link_pred`"""
        pass

    @abstractmethod
    def list_of_metrics_names(self) -> List[str]:
        pass

    @staticmethod
    def _set_running_args(parser: ArgumentParser) -> ArgumentParser:
        parser =  EdgeBankTrainer._set_running_args(parser)
        parser.add_argument('--k-value', type=int, help='k_value for computing ranking metrics', default=10)
        parser.add_argument('--load-neg-samples', action='store_true')
        parser.add_argument('--visualize', action='store_true')
        return parser
    
    def _get_visualization_dir(self) -> str:
        return os.path.join(self.run_dir,
            "vis",
        )
    
    def _get_run_save_dir(self) -> str:
        dtdgtrainer_save_run_dir = super(LinkPredEdgeBankTrainer, self)._get_run_save_dir()

        return os.path.join(
            dtdgtrainer_save_run_dir, 
            "linkpred",
            self.args.data)
    
    def create_data_loaders(self) -> Tuple[torchDataLoader, torchDataLoader, torchDataLoader, torchDataLoader]:
        if not hasattr(self.args, "node_feat"):
            setattr(self.args, "node_feat", NodeFeatType.CONSTANT)
        train_dataset = LinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), self.args.data, split="train", node_feat=self.args.node_feat)
        val_dataset = LinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), self.args.data, split="val", node_feat=self.args.node_feat)
        test_dataset = LinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), self.args.data, split="test", node_feat=self.args.node_feat)
        test_inductive_dataset = LinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), self.args.data, split="test_inductive", node_feat=self.args.node_feat)
        
        # During link prediction training, each snapshot is given as batch data
        # No shuffling should happen as it is important to keep the order of snapshots.
        train_loader = torchDataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = torchDataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = torchDataLoader(test_dataset, batch_size=1, shuffle=False)
        test_inductive_loader = torchDataLoader(test_inductive_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader, test_inductive_loader
    
    @abstractmethod
    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):
        pass


    def get_pos_link(self, curr_snapshot: torch.Tensor):
        # Separate positive and negative pairs
        pos_src, pos_dst = torch.nonzero(curr_snapshot, as_tuple=True)
        pos_src = np.array(pos_src)
        pos_dst = np.array(pos_dst)
        return pos_src, pos_dst
    
    def _eval_predict_current_timestep(self, curr_snapshot: torch.Tensor):
        pos_src, pos_dst = self.get_pos_link(curr_snapshot)
        out_2d = torch.zeros_like(curr_snapshot).cpu()
        
        # To prevent considering self-loop edges as negative pairs, set diagonal elements as zero.
        curr_snapshot.fill_diagonal_(1)
        neg_src, neg_dst = torch.nonzero(curr_snapshot == 0, as_tuple=True)
        curr_snapshot.fill_diagonal_(0)
        
        pos_pred = self.model['edgebank'].predict_link(pos_src, pos_dst)
        neg_pred = self.model['edgebank'].predict_link(neg_src.cpu().numpy(), neg_dst.cpu().numpy())
        out_2d.fill_(torch.nan)
        out_2d[pos_src, pos_dst] = torch.from_numpy(pos_pred).float()
        out_2d[neg_src, neg_dst] = torch.from_numpy(neg_pred).float()
        out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

        # Valid number assertion
        assert torch.all(torch.isfinite(out_2d))

        return out_2d

    def eval(self, split_mode: str):
        # loading the validation and test negative samples
        if self.args.load_neg_samples:
            self.neg_sampler.load_eval_set(os.path.join(self.args.root_load_save_dir, self.args.data_loc, self.args.data, f"{split_mode}_ns.pkl"), split_mode=split_mode)

        if split_mode == "val":
            eval_loader = self.val_loader
        elif split_mode == "test":
            eval_loader = self.test_loader
        elif split_mode == "test_inductive":
            eval_loader = self.test_inductive_loader
        else:
            raise NotImplementedError()

        metrics_list = {k: [] for k in self.list_of_metrics_names()}

        for snapshot_idx, batch in enumerate(eval_loader):
            prev_snapshot, _, curr_snapshot, _, node_feat, snapshot_t = batch
            assert len(prev_snapshot) == 1, "The batch size for this task should be one."
            prev_snapshot = prev_snapshot[0]
            curr_snapshot = curr_snapshot[0]
            node_feat = node_feat[0]
            snapshot_t = snapshot_t[0]

            out_2d = self._eval_predict_current_timestep(curr_snapshot.float())
            
            if self.args.visualize:
                # Visualize outputs during evaluation mode
                self.vis(out_2d, 
                        curr_snapshot, 
                        split_mode, 
                        'None', 
                        str(snapshot_idx))

            # compute performance metrics
            self.update_metrics(curr_snapshot, snapshot_t, snapshot_idx, metrics_list, split_mode)

        # Take the average of all metrics among snapshots
        metrics_list = {metric: np.nan_to_num(np.mean(list_of_values), nan=-1.0) for metric, list_of_values in metrics_list.items()}
        return metrics_list
    