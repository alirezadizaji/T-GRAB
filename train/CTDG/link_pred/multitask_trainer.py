from abc import abstractmethod
from argparse import ArgumentParser
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as torchDataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from ....dataset.CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from ....dataset.DTDG.torch_dataset.link_pred.node_feat_static import LinkPredNodeFeatureStaticDataset
from ....model.link_predictor import LinkPredictor
from ....utils import visualizer
from ...CTDG.trainer import CTDGTrainer, NODE_EMB_MODEL_NAME
from ....model.node_emb import NodeEmbeddingModel

class MultiTaskLinkPredTrainer(CTDGTrainer):
    def __init__(self):
        super(MultiTaskLinkPredTrainer, self).__init__()
        assert NODE_EMB_MODEL_NAME in self.model.keys(), f'self.model should contain a graph learning model, passed as {NODE_EMB_MODEL_NAME} keyword.'
        assert 'link_pred' in self.model.keys(), 'self.model should contain a link-prediction head, passed as `link_pred` keyword.'
        assert isinstance(self.model[NODE_EMB_MODEL_NAME], NodeEmbeddingModel), 'gnn model should be a child class of GraphModel.'
        assert self.val_loader.batch_size == 1, "Validation data loader should have batch size of 1."
        assert self.test_loader.batch_size == 1, "Test data loader should have batch size of 1."

        self.task_list = [
            ("periodicity", "(2, 256)/fixed_er-100n-40trW-1vW-1tsW-p0.0-fp0.01"),
            ("memory_node", "(1, 1000)/memory_node-1001n-erpm-0.1vr-0.1tr-0.1tir-0.1tinnr-0.0001ep-0.001epi"),
            # ("long_range", "(16, 100)/long_range-100n-erpm-0.1vr-0.1tr-0.01eps-0.4epp"),
        ]
 
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
        

    def list_of_metrics_names(self) -> List[str]:
        return ["loss"]

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
        parser =  CTDGTrainer._set_running_args(parser)
        parser.add_argument('--num-epochs-to-visualize', type=int, default=1, required=False, help='Number of epochs to visualize output generation during evaluation of training phase.')
        parser.add_argument('--train-batch-size', type=int, default=1)
        parser.add_argument('--task-loading-mode', type=str, default="sequential", choices=["sequential", "parallel"], required=False, help='Whether to load tasks sequentially or in parallel.')
        return parser
    
    def _get_visualization_dir(self) -> str:
        return os.path.join(self.run_dir,
            "vis",
        )
    
    def _get_run_save_dir(self) -> str:
        ctdgtrainer_save_run_dir = super(LinkPredTrainer, self)._get_run_save_dir()

        return os.path.join(
            ctdgtrainer_save_run_dir, 
            "linkpred",
            self.args.data,
            f"snapshot_training=True,batch_size={self.args.train_batch_size},task_loading_mode={self.args.task_loading_mode}")
    
    def create_data_loaders(self) -> Tuple[torchDataLoader, torchDataLoader, torchDataLoader]:
        split_multi_task_datasets = {}

        for split in ["train", "val", "test", "test_inductive"]:
            datasets = []
            for (task_name, dataset_pattern) in self.task_list:
                dataset = LinkPredNodeFeatureStaticDataset(
                                    os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                                    dataset_pattern, 
                                    split=split, 
                                    node_feat=self.args.node_feat, 
                                    node_feat_dim=self.args.node_feat_dim,
                                    to_dense=False,
                                return_edge_info=True)

                datasets.append((task_name, dataset))
        
            if self.args.task_loading_mode == "sequential":
                multi_task_dataset = SequentialMultiTaskDataset(datasets)
            elif self.args.task_loading_mode == "parallel":
                multi_task_dataset = ParallelMultiTaskDataset(datasets)
            else:
                raise NotImplementedError()

            split_multi_task_datasets[split] = multi_task_dataset

        train_loader = torchDataLoader(split_multi_task_datasets["train"], batch_size=self.args.train_batch_size, shuffle=False)
        val_loader = torchDataLoader(split_multi_task_datasets["val"], batch_size=1, shuffle=False)
        test_loader = torchDataLoader(split_multi_task_datasets["test"], batch_size=1, shuffle=False)
        test_inductive_loader = torchDataLoader(split_multi_task_datasets["test_inductive"], batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader, test_inductive_loader
    
    @abstractmethod
    def before_epoch_training(self):
        pass
    
    @abstractmethod
    def after_iteration_training(self):
        pass

    @abstractmethod
    def after_epoch_training(self):
        pass

    @abstractmethod
    def before_epoch_evaluation(self, split_mode: str):
        pass
    
    @abstractmethod
    def after_iteration_evaluation(self, split_mode):
        pass

    @abstractmethod
    def after_epoch_evaluation(self, split_mode: str):
        pass

    @abstractmethod
    def forward_backbone(self, 
                batch_src: torch.Tensor, 
                batch_dst: torch.Tensor, 
                batch_t: torch.Tensor, 
                batch_edge_id: torch.Tensor, 
                batch_edge_feat: torch.Tensor,
                batch_neg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None,
                ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the backbone model to generate node embeddings for link prediction.

        Args:
            batch_src (torch.Tensor): Source node indices for positive links
            batch_dst (torch.Tensor): Destination node indices for positive links  
            batch_t (torch.Tensor): Timestamps for the links
            batch_edge_id (torch.Tensor): Edge IDs for the links
            batch_edge_feat (torch.Tensor): Edge features for the links
            batch_neg (Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], optional): 
                Tuple of (src, dst, t) tensors for negative samples. Defaults to None.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: A tuple containing:
                - Tuple of (src_emb, dst_emb) for positive samples
                - Tuple of (src_emb, dst_emb) for negative samples
                Where each embedding has shape [batch_size, embedding_dim]
        """

        pass
    
    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str,
                        dataset: Dataset):
        pass
    
    def get_neg_link(self, pos_src: torch.Tensor, pos_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            device=self.device
        
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
    

    def _train_for_one_epoch_snapshot_based(self):
        self.before_epoch_training()

        self.model[NODE_EMB_MODEL_NAME].train()
        self.model['link_pred'].train()
        train_losses = []

        # Each batch represents only one snapshot.
        for batch_idx, batch in enumerate(self.train_loader):
            batch_mask, batch_task_name, batch_task_dataset = batch
            print(f"\t\%\% Training iteration {batch_idx} out of {len(self.train_loader)}", flush=True)
            self.optim.zero_grad()
            
            # `mask` is created within the collate function.
            # `mask` enables to pass multiple snapshots in one batch, as it is highly possible that each one can have different number of real links.
            (pos_src, pos_dst), _, curr_t, pos_edge_feat, pos_edge_ids, mask = batch
            seq_len = pos_src.shape[1]
            curr_t = curr_t.unsqueeze(1).repeat(1, seq_len)
            pos_src = pos_src[mask].to(self.device)
            pos_dst = pos_dst[mask].to(self.device)
            pos_edge_feat = pos_edge_feat[mask].to(self.device)
            pos_edge_ids = pos_edge_ids[mask].to(self.device)
            pos_t = curr_t[mask].to(self.device)
            
            neg_src, neg_dst = self.get_neg_link(pos_src, pos_dst)
            neg_t = pos_t.clone()
            assert pos_src.size() == neg_src.size(), f"Number of negative links ({neg_src.size()}) is not the same as positive ones ({pos_src.size()}."
            # assert curr_t.unique().numel() == 1, "All timestamps should be the same."
            
            # Forward model that is behind an MLP model to encode node embeddings. 
            (pos_src_node_embeddings, pos_dst_node_embeddings), (neg_src_node_embeddings, neg_dst_node_embeddings) = \
                    self.forward_backbone(pos_src, 
                                        pos_dst, 
                                        pos_t,
                                        batch_edge_id=pos_edge_ids, 
                                        batch_edge_feat=pos_edge_feat,
                                        batch_neg=(neg_src, neg_dst, neg_t))
            
            assert pos_src_node_embeddings.shape[0] == pos_src.shape[0], f"Mistmatch size between backbone output ({pos_src_node_embeddings.size()}) and input ({pos_src.size()})."
            assert pos_src_node_embeddings.shape[0] == pos_src.shape[0], f"Mistmatch size between backbone output ({pos_src_node_embeddings.size()}) and input ({pos_src.size()})."

            # forward link prediction model
            pos_pred: torch.Tensor = self.model['link_pred'](pos_src_node_embeddings, pos_dst_node_embeddings)
            neg_pred: torch.Tensor = self.model['link_pred'](neg_src_node_embeddings, neg_dst_node_embeddings)

            assert torch.all(pos_pred <= 1) and torch.all(pos_pred >= 0), "Make sure predictions are in range [0, 1]."
            assert pos_src_node_embeddings.shape[0] == pos_src.shape[0], f"Mistmatch size between backbone output ({pos_src_node_embeddings.size()}) and input ({pos_src.size()})."


            # Loss computation
            loss = self.criterion(pos_pred, torch.ones_like(pos_pred))
            loss = loss + self.criterion(neg_pred, torch.zeros_like(neg_pred))
            loss.backward()
            self.optim.step()
            train_losses.append(loss.detach().item())

            self.after_iteration_training()

        self.after_epoch_training()

        avg_loss = np.mean(train_losses)
        print(f"Epoch: {self.epoch:02d}, Loss: {avg_loss:.4f}.")
        perf_metrics = {
            "loss": avg_loss,
        }
       
        return perf_metrics


    def train_for_one_epoch(self):
        return self._train_for_one_epoch_snapshot_based()
        

    def _eval_predict_current_timestep(self, curr_snapshot: torch.Tensor, snapshot_t: int, dataset: Dataset):
        pos_src, pos_dst = torch.nonzero(curr_snapshot, as_tuple=True)
        pos_t = torch.full_like(pos_src, fill_value=snapshot_t)
        pos_edge_ids, pos_edge_feats = dataset.get_attr(pos_src, 
                                                        pos_dst, 
                                                        pos_t, attrs=["edge_ids", "edge_feat"])
        neg_src, neg_dst = torch.nonzero(curr_snapshot == 0, as_tuple=True)
        neg_t = torch.full_like(neg_src, fill_value=snapshot_t)
        (pos_src_node_embeddings, pos_dst_node_embeddings), (neg_src_node_embeddings, neg_dst_node_embeddings) = \
                        self.forward_backbone(pos_src, 
                                            pos_dst, 
                                            pos_t,
                                            batch_edge_id=pos_edge_ids,
                                            batch_edge_feat=pos_edge_feats,
                                            batch_neg=(neg_src, neg_dst, neg_t))
        
        out_2d = torch.zeros_like(curr_snapshot).cpu()
        
        pos_pred = self.model['link_pred'](pos_src_node_embeddings, pos_dst_node_embeddings)
        neg_pred = self.model['link_pred'](neg_src_node_embeddings, neg_dst_node_embeddings)

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
        self.before_epoch_evaluation(split_mode)

        assert not torch.is_grad_enabled(), "During evaluation, torch grad should be disabled."

        self.model[NODE_EMB_MODEL_NAME].eval()
        self.model['link_pred'].eval()

        if split_mode == 'val':
            eval_loader = self.val_loader
        elif split_mode == 'test':
            eval_loader = self.test_loader
        elif split_mode == 'test_inductive':
            eval_loader = self.test_inductive_loader
        else:
            raise NotImplementedError()

        metrics_list = {k: [] for k in self.list_of_metrics_names()}

        for snapshot_idx, batch in enumerate(eval_loader):
            print(f"\t\%\% Evaluation iteration {snapshot_idx} out of {len(eval_loader)}", flush=True)
            prev_snapshot, _, curr_snapshot, _, node_feat, snapshot_t = batch
            del prev_snapshot
            assert len(curr_snapshot) == 1, "The batch size for this task should be one."
            curr_snapshot = curr_snapshot[0]
            node_feat = node_feat[0]
            snapshot_t = snapshot_t[0]

            curr_snapshot = curr_snapshot.to(self.device)

            if self.args.num_epochs_to_visualize > 0:
                if (self.args.eval_mode or self.epoch in self.epochs_to_visualize):
                    out_2d = self._eval_predict_current_timestep(curr_snapshot.float(), snapshot_t, eval_loader.dataset)
                    self._visualize_target_and_prediction(out_2d, 
                                                        curr_snapshot, 
                                                        split_mode, 
                                                        snapshot_idx,
                                                        self.epoch)                

            # compute performance metrics
            self.update_metrics(curr_snapshot, snapshot_t, snapshot_idx, metrics_list, split_mode, eval_loader.dataset)
            
            self.after_iteration_evaluation(split_mode)

        self.after_epoch_evaluation(split_mode)

        # Take the average of all metrics among snapshots
        metrics_list = {metric: np.nan_to_num(np.mean(list_of_values), nan=-1.0) for metric, list_of_values in metrics_list.items()}
        return metrics_list

    def early_stopping_checker(self, early_stopper) -> bool:
        if early_stopper.step_check(self.train_perf_list["loss"][-1], self.model, op_to_cont="dec"):
            return True
        return False