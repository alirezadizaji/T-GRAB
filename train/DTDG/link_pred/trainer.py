from abc import abstractmethod
from argparse import ArgumentParser
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as torchDataLoader

from ....dataset.DTDG.torch_dataset.link_pred.node_feat_static import LinkPredNodeFeatureStaticDataset
from ....model.link_predictor import LinkPredictor
from ....utils import visualizer
from ...DTDG.trainer import DTDGTrainer, NODE_EMB_MODEL_NAME
from ....model.node_emb import NodeEmbeddingModel

class LinkPredTrainer(DTDGTrainer):
    def __init__(self):
        super(LinkPredTrainer, self).__init__()
        assert NODE_EMB_MODEL_NAME in self.model.keys(), f'self.model should contain a graph learning model, passed as {NODE_EMB_MODEL_NAME} keyword.'
        assert 'link_pred' in self.model.keys(), 'self.model should contain a link-prediction head, passed as `link_pred` keyword.'
        assert isinstance(self.model[NODE_EMB_MODEL_NAME], NodeEmbeddingModel), 'gnn model should be a child class of GraphModel.'
        assert self.train_loader.batch_size == 1, "Train data loader should have batch size of 1."
        assert self.val_loader.batch_size == 1, "Validation data loader should have batch size of 1."
        assert self.test_loader.batch_size == 1, "Test data loader should have batch size of 1."

        # Check the dataset name follows the regex pattern
        dataset_regex_pattern, dataset_name_part_to_check = self.get_dataset_regex_pattern()
        import re
        assert re.match(dataset_regex_pattern, dataset_name_part_to_check), f"Given dataset name ({self.args.data}) doesn't match with the dataset regex pattern ({dataset_regex_pattern})."
 
        # Visualizer to output graph generation by link prediction
        self.vis = visualizer(save_dir=self._get_visualization_dir(), num_nodes=self.train_loader.dataset.num_nodes, node_pos=self.args.node_pos)

        # Determine which epochs to visualize the output generation during evaluation of training phase.
        if self.args.num_epochs_to_visualize > 0:
            num_epochs_to_visualize = min(self.args.num_epochs_to_visualize, self.args.num_epoch)
            visualize_number = np.arange(num_epochs_to_visualize) + 1
            visualize_interval_size = int(self.args.num_epoch / num_epochs_to_visualize)
            visualize_interval_size = max(visualize_interval_size, 1)
            epochs_to_visualize = visualize_number * visualize_interval_size
            self.epochs_to_visualize = set(np.clip(epochs_to_visualize, 0, a_max=self.args.num_epoch - 1))
            if num_epochs_to_visualize == 1:
                assert len(self.epochs_to_visualize) == 1 and (self.args.num_epoch - 1) in self.epochs_to_visualize, \
                    f"For one-time visualization, the epoch to do this should be the last one ({self.args.num_epoch - 1}); got {self.epochs_to_visualize} instead."
        else:
            self.epochs_to_visualize = []
        
    @abstractmethod
    def get_dataset_regex_pattern(self) -> str:
        """ Each task in link prediction supports a group of datasets that follow a regex pattern. To understand current supported tasks,
        please checkout subfolders of `link_pred`"""
        ...

    @abstractmethod
    def list_of_metrics_names(self) -> List[str]:
        pass

    @property
    def parameters(self):
        return set(self.model[NODE_EMB_MODEL_NAME].parameters()) | set(self.model['link_pred'].parameters())
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters,
            lr=self.args.lr,
        )
    
    def get_criterion(self) -> torch.nn.Module:
        return torch.nn.BCELoss()

    def get_model(self) -> Dict[str, torch.nn.Module]:
        models = super(LinkPredTrainer, self).get_model()
        node_emb_model: 'NodeEmbeddingModel' = models[NODE_EMB_MODEL_NAME]
        print(f"@@@ The class of node embedding model: {type(node_emb_model)} @@@", flush=True)
        link_pred = LinkPredictor(node_emb_model.out_dimension, num_layers=2)
        link_pred.to(self.device)

        models['link_pred'] = link_pred

        return models
    
    @staticmethod
    def _set_running_args(parser: ArgumentParser) -> ArgumentParser:
        parser =  DTDGTrainer._set_running_args(parser)
        parser.add_argument('--num-epochs-to-visualize', type=int, default=1, required=False, help='Number of epochs to visualize output generation during evaluation of training phase.')
        parser.add_argument('--back-prop-window-size', type=int, default=1)
        parser.add_argument('--loss-computation', choices=['backward_only_last', 'backward_sum'], default='backward_only_last')
        return parser
    
    def _get_visualization_dir(self) -> str:
        return os.path.join(self.run_dir,
            "vis",
        )
    
    def _get_run_save_dir(self) -> str:
        dtdgtrainer_save_run_dir = super(LinkPredTrainer, self)._get_run_save_dir()

        return os.path.join(
            dtdgtrainer_save_run_dir, 
            "linkpred",
            self.args.data)
    
    def create_data_loaders(self) -> Tuple[torchDataLoader, torchDataLoader, torchDataLoader]:
        train_dataset = LinkPredNodeFeatureStaticDataset(
                            os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                            self.args.data, 
                            split="train", 
                            node_feat=self.args.node_feat, 
                            node_feat_dim=self.args.node_feat_dim)
        val_dataset = LinkPredNodeFeatureStaticDataset(
                            os.path.join(self.args.root_load_save_dir, 
                            self.args.data_loc), 
                            self.args.data, 
                            split="val", 
                            node_feat=self.args.node_feat, 
                            node_feat_dim=self.args.node_feat_dim)
        test_dataset = LinkPredNodeFeatureStaticDataset(
                            os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                            self.args.data, 
                            split="test", 
                            node_feat=self.args.node_feat, 
                            node_feat_dim=self.args.node_feat_dim)
        test_inductive_dataset = LinkPredNodeFeatureStaticDataset(
                            os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                            self.args.data, 
                            split="test_inductive", 
                            node_feat=self.args.node_feat, 
                            node_feat_dim=self.args.node_feat_dim)

        # During link prediction training, each snapshot is given as batch data
        # No shuffling should happen as it is important to keep the order of snapshots.
        train_loader = torchDataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = torchDataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = torchDataLoader(test_dataset, batch_size=1, shuffle=False)
        test_inductive_loader = torchDataLoader(test_inductive_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader, test_inductive_loader
    
    @abstractmethod
    def before_training(self):
        pass

    @abstractmethod
    def forward_backbone(self, snapshot: torch.Tensor, edge_feat_snapshot: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        """ This function forwards the input through the model appeared behind an MLP. The output is an embedding representations of the nodes in the snapshot """
        pass
    
    @abstractmethod
    def before_starting_window_training(self):
        pass

    @abstractmethod
    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):
        pass

    def _get_model_card(self) -> str:
        """ In Discrete-time dynamic graph training, type of node feature and its dimension is considered within the model card."""
        dtdg_model_card = super(LinkPredTrainer, self)._get_model_card()
        
        return f"window_size={self.args.back_prop_window_size}_loss_computation={self.args.loss_computation}_" + dtdg_model_card
    
    def get_pos_link(self, curr_snapshot: torch.Tensor):
        # Separate positive and negative pairs
        pos_src, pos_dst = torch.nonzero(curr_snapshot, as_tuple=True)

        return pos_src, pos_dst
    
    def get_neg_link(self, pos_src: torch.Tensor, pos_dst: torch.Tensor, generator:Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates negative links by random sampling, ensuring no collisions with positive links.

        For each positive source node, randomly samples a destination node to create a negative link.
        Uses recursive sampling to ensure no overlap between positive and negative links.

        Args:
            pos_src (torch.Tensor): Source nodes from positive links
            pos_dst (torch.Tensor): Destination nodes from positive links

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - neg_src: Source nodes for negative links
                - neg_dst: Destination nodes for negative links
            
        Note:
            The method guarantees that returned negative links do not exist in the positive links
            by recursively resampling any colliding pairs until all negative samples are unique.
        """
        neg_src = pos_src.clone()

        src_dst_pairs = {(src, dst): True for src, dst in zip(pos_src, pos_dst)}

        neg_dst = torch.randint(
            0,
            self.train_loader.dataset.num_nodes,
            size=(pos_dst.numel(),),
            dtype=torch.long,
            device=self.device,
            generator=generator
        )
        collision_mask = torch.tensor([src_dst_pairs.get((_src, _dst), False) == True for _src, _dst in zip(neg_src, neg_dst)])

        # Stop random negative sampling if no collision is found.
        if (~collision_mask).all():
            return neg_src, neg_dst
        
        # Recursively continue to get negative links until no collision is found.
        neg_src_non_collision, neg_dst_non_collision = neg_src[~collision_mask], neg_dst[~collision_mask]
        neg_src_collision, neg_dst_collision = self.get_neg_link(pos_src[collision_mask], pos_dst[collision_mask])
            
        neg_src = torch.cat([neg_src_non_collision, neg_src_collision])
        neg_dst = torch.cat([neg_dst_non_collision, neg_dst_collision])
        
        assert pos_src.size() == neg_src.size(), f"Number of negative links ({neg_src.size()}) is not the same as positive ones ({pos_src.size()}."
        collision_mask = torch.tensor([src_dst_pairs.get((_src, _dst), False) == True for _src, _dst in zip(neg_src, neg_dst)])
        assert (~collision_mask).all(), "There should be no collision between positive and negative links."
        
        return neg_src, neg_dst 

    @property
    def retain_graph(self) -> bool:
        return False
    
    def train_for_one_epoch(self):
        self.before_training()

        train_losses = []
        window_losses: torch.Tensor = 0.0

        metrics_list = {k: [] for k in self.list_of_metrics_names()}

        # Each batch represents only one snapshot.
        num_batches = len(self.train_loader)
        num_batches_to_log = 4
        batches_chunk = num_batches // num_batches_to_log
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx % batches_chunk == 0:
                print(f"\t\%\% Training iteration {batch_idx} out of {num_batches}", flush=True)
            
            if batch_idx % self.args.back_prop_window_size == 0:
                self.before_starting_window_training()
                self.optim.zero_grad()
                window_losses = 0.0
            
            prev_snapshot, prev_edge_feat_snapshot, curr_snapshot, curr_edge_feat_snapshot, node_feat, snapshot_t = batch
            assert len(prev_snapshot) == 1, "This task supports only batch 1 (single-snapshot) training."
            prev_snapshot = prev_snapshot[0]
            if batch_idx == 0:
                # For training epochs > starting_epoch, prev_snapshot is recorded at the end of evaluation. This helps passing the right sequence without interrupting.
                if hasattr(self, 'prev_snapshot'):
                    prev_snapshot = self.prev_snapshot.float()
                else:
                    prev_snapshot = torch.rand_like(prev_snapshot.float())

            prev_edge_feat_snapshot = prev_edge_feat_snapshot[0]
            curr_edge_feat_snapshot = curr_edge_feat_snapshot[0]
            curr_snapshot = curr_snapshot[0]
            node_feat = node_feat[0]
            snapshot_t = snapshot_t[0]
            prev_snapshot = prev_snapshot.to(self.device)
            prev_edge_feat_snapshot = prev_edge_feat_snapshot.to(self.device)
            curr_snapshot = curr_snapshot.to(self.device)
            node_feat = node_feat.to(self.device)

            pos_src, pos_dst = self.get_pos_link(curr_snapshot)
            neg_src, neg_dst = self.get_neg_link(pos_src, pos_dst)
            assert pos_src.size() == neg_src.size(), f"Number of negative links ({neg_src.size()}) is not the same as positive ones ({pos_src.size()}."

            # Forward model that is behind an MLP model to encode node embeddings. 
            z = self.forward_backbone(prev_snapshot, prev_edge_feat_snapshot, node_feat)
            self.z = z
            assert self.z.shape[0] == curr_snapshot.shape[0], f"Graph embedding with shape `{self.z.shape}` has not same number of nodes as current snapshot `{curr_snapshot.shape}`."

            # forward link prediction model
            pos_pred: torch.Tensor = self.model['link_pred'](z[pos_src], z[pos_dst])
            neg_pred: torch.Tensor = self.model['link_pred'](z[neg_src], z[neg_dst])
            
            # compute performance metrics
            self.update_metrics(curr_snapshot, snapshot_t, batch_idx, metrics_list, split_mode="train")
            
            assert torch.all(pos_pred <= 1) and torch.all(pos_pred >= 0), "Make sure predictions are in range [0, 1]."

            # Loss computation
            loss_t = self.criterion(pos_pred, torch.ones_like(pos_pred))
            loss_t = loss_t + self.criterion(neg_pred, torch.zeros_like(neg_pred))
            window_losses = window_losses + loss_t
            
            # Backpropagation
            if (batch_idx + 1) % self.args.back_prop_window_size == 0:
                if self.args.loss_computation == 'backward_only_last':
                    window_losses = loss_t
                if batch_idx == len(self.train_loader) - 1: # Make sure to release the computational graph at the last batch
                    window_losses.backward()                
                else:
                    window_losses.backward(retain_graph=self.retain_graph)
                self.optim.step()
                train_losses.append(window_losses.detach().item())

        # Last forward before going to the evaluation phase.
        curr_edge_feat_snapshot = curr_edge_feat_snapshot.to(self.device)
        z = self.forward_backbone(curr_snapshot, curr_edge_feat_snapshot, node_feat)
        self.z = z
        self.prev_snapshot = curr_snapshot

        # Metrics computation
        metrics_list = {metric: np.nan_to_num(np.mean(list_of_values), nan=0.0) for metric, list_of_values in metrics_list.items()}
        avg_loss = np.mean(train_losses)
        print(f"Epoch: {self.epoch:02d}, Loss: {avg_loss:.4f}, train {self.val_first_metric}: {metrics_list[self.val_first_metric]:.2f}.")
        perf_metrics = {
            "loss": avg_loss,
            self.val_first_metric: metrics_list[self.val_first_metric],
        }
       
        return perf_metrics


    def _eval_predict_current_timestep(self, curr_snapshot: torch.Tensor):
        z = self.z

        pos_src, pos_dst = self.get_pos_link(curr_snapshot)
        out_2d = torch.zeros_like(curr_snapshot).cpu()
        
        # To prevent considering self-loop edges as negative pairs, set diagonal elements as zero.
        curr_snapshot.fill_diagonal_(1)
        neg_src, neg_dst = torch.nonzero(curr_snapshot == 0, as_tuple=True)
        curr_snapshot.fill_diagonal_(0)
        
        pos_pred = self.model['link_pred'](z[pos_src], z[pos_dst])
        neg_pred = self.model['link_pred'](z[neg_src], z[neg_dst])

        out_2d.fill_(torch.nan)
        out_2d[pos_src, pos_dst] = pos_pred.squeeze(-1).detach().cpu()
        out_2d[neg_src, neg_dst] = neg_pred.squeeze(-1).detach().cpu()
        out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

        # Valid number assertion
        assert torch.all(torch.isfinite(out_2d))

        return out_2d

    def _visualize_target_and_prediction(self, out_2d, curr_snapshot, split_mode, snapshot_idx, epoch) -> None:
        self.vis(out_2d, 
                curr_snapshot, 
                split_mode, 
                epoch, 
                str(snapshot_idx))

    def eval_for_one_epoch(self, split_mode: str):
        assert not torch.is_grad_enabled(), "During evaluation, torch grad should be disabled."

        if split_mode == 'train':
            eval_loader = self.train_loader
        elif split_mode == 'val':
            eval_loader = self.val_loader
        elif split_mode == 'test':
            eval_loader = self.test_loader
        elif split_mode == 'test_inductive':
            eval_loader = self.test_inductive_loader
        else:
            raise NotImplementedError()

        metrics_list = {k: [] for k in self.list_of_metrics_names()}

        num_batches = len(eval_loader)
        num_batches_to_log = 4
        batches_chunk = num_batches // num_batches_to_log
        
        for snapshot_idx, batch in enumerate(eval_loader):
            if snapshot_idx % batches_chunk == 0:
                print(f"\t\%\% Evaluation iteration {snapshot_idx} out of {num_batches}", flush=True)
            
            _, _, curr_snapshot, curr_edge_feat_snapshot, node_feat, snapshot_t = batch
            assert len(curr_snapshot) == 1, "The batch size for this task should be one."
            curr_snapshot = curr_snapshot[0]
            curr_edge_feat_snapshot = curr_edge_feat_snapshot[0]
            node_feat = node_feat[0]
            snapshot_t = snapshot_t[0]

            curr_snapshot = curr_snapshot.to(self.device)
            curr_edge_feat_snapshot = curr_edge_feat_snapshot.to(self.device)
            node_feat = node_feat.to(self.device)

            # In evaluation mode, at first step there is no "previous snapshot" to input. Instead, use current snapshot as input.
            if not hasattr(self, 'z'):
                z = self.forward_backbone(curr_snapshot.float(), curr_edge_feat_snapshot, node_feat)
                self.z = z
            
            # Visualize outputs during evaluation mode
            if self.args.num_epochs_to_visualize:
                if (self.args.eval_mode or self.epoch in self.epochs_to_visualize):
                    out_2d = self._eval_predict_current_timestep(curr_snapshot.float())
                    self._visualize_target_and_prediction(out_2d, 
                                                        curr_snapshot, 
                                                        split_mode, 
                                                        snapshot_idx,
                                                        self.epoch)                

            # compute performance metrics
            self.update_metrics(curr_snapshot, snapshot_t, snapshot_idx, metrics_list, split_mode)

            z = self.forward_backbone(curr_snapshot.float(), curr_edge_feat_snapshot, node_feat)
            self.z = z
            assert self.z.shape[0] == curr_snapshot.shape[0], f"Graph embedding with shape `{self.z.shape}` has not same number of nodes as current snapshot `{curr_snapshot.shape}`."
            
            # Keep the last snapshot seen during the evaluation. The first step of training phase will need it.
            self.prev_snapshot = curr_snapshot

        # Take the average of all metrics among snapshots
        metrics_list = {metric: np.nan_to_num(np.mean(list_of_values), nan=-1.0) for metric, list_of_values in metrics_list.items()}
        return metrics_list

    def early_stopping_checker(self, early_stopper) -> bool:
        if early_stopper.step_check(self.train_perf_list["loss"][-1], self.model, op_to_cont="dec"):
            return True
        return False 