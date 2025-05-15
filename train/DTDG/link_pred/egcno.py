import numpy as np
import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.egcno import EvolveGCNTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredEvolveGCNTrainer(LinkPredTrainer, EvolveGCNTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    def forward_node_emb_model_for_one_iter(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        src, dst = torch.nonzero(snapshot, as_tuple=True)    
        edge_index = torch.stack([src, dst], dim=0)
        edge_feat = snapshot_feat[src, dst]
        z = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                edge_index.long(),
                edge_weight=edge_feat)
        
        return z


    def train_for_one_epoch(self):
        self.before_training()

        self.model[NODE_EMB_MODEL_NAME].train()
        self.model['link_pred'].train()
        loss_t = 0.0

        for unit in self.model[NODE_EMB_MODEL_NAME].units:
            unit.weight = None
       
        metrics_list = {k: [] for k in self.list_of_metrics_names()}

        # Each batch represents only one snapshot.
        for batch_idx, batch in enumerate(self.train_loader):
            print(f"\t\%\% Training iteration {batch_idx} out of {len(self.train_loader)}", flush=True)
            self.optim.zero_grad()
            
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
            z = self.forward_node_emb_model_for_one_iter(prev_snapshot, prev_edge_feat_snapshot, node_feat)
            self.z = z
            assert self.z.shape[0] == curr_snapshot.shape[0], f"Graph embedding with shape `{self.z.shape}` has not same number of nodes as current snapshot `{curr_snapshot.shape}`."

            # forward link prediction model
            pos_pred: torch.Tensor = self.model['link_pred'](z[pos_src], z[pos_dst])
            neg_pred: torch.Tensor = self.model['link_pred'](z[neg_src], z[neg_dst])
            
            # compute performance metrics
            self.update_metrics(curr_snapshot, snapshot_t, batch_idx, metrics_list, split_mode="train")
            
            assert torch.all(pos_pred <= 1) and torch.all(pos_pred >= 0), "Make sure predictions are in range [0, 1]."

            # Loss computation
            loss_t = loss_t + self.criterion(pos_pred, torch.ones_like(pos_pred))
            loss_t = loss_t + self.criterion(neg_pred, torch.zeros_like(neg_pred))
            
            loss_t.backward()
            self.optim.step()

            for unit in self.model[NODE_EMB_MODEL_NAME].units:
                unit.initial_weight = unit.weight.detach()
                unit.weight = None

        # Last forward before going to the evaluation phase.
        curr_edge_feat_snapshot = curr_edge_feat_snapshot.to(self.device)
        z = self.forward_node_emb_model_for_one_iter(curr_snapshot, curr_edge_feat_snapshot, node_feat)
        self.z = z
        self.prev_snapshot = curr_snapshot

        # Metrics computation
        metrics_list = {metric: np.nan_to_num(np.mean(list_of_values), nan=0.0) for metric, list_of_values in metrics_list.items()}
        print(f"Epoch: {self.epoch:02d}, Loss: {avg_loss.item():.4f}, train {self.val_first_metric}: {metrics_list[self.val_first_metric]:.2f}.")
        perf_metrics = {
            "loss": avg_loss.item(),
            self.val_first_metric: metrics_list[self.val_first_metric],
        }
       
        return perf_metrics
    
