"""
2nd edition of NegativeEdgeGenerator
---
This is an editted version of `NegativeEdgeGenerator` implemented by TGB github. 
The goal is to support full negative edge sampling.
"""

import os
from tqdm import tqdm

import numpy as np
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl
from tgb.linkproppred.negative_generator import NegativeEdgeGenerator



class NegativeEdgeGeneratorV2(NegativeEdgeGenerator):
    def generate_negative_samples_rnd(self, 
                                    data: TemporalData, 
                                    split_mode: str, 
                                    filename: str,
                                    ) -> None:
        r"""
        Generate negative samples based on the `HIST-RND` strategy:
            - for each positive edge, sample a batch of negative edges from all possible edges with the same source node
            - filter actual positive edges
        
        Parameters:
            data: an object containing positive edges information
            split_mode: specifies whether to generate negative edges for 'validation' or 'test' splits
            filename: name of the file containing the generated negative edges
        """
        print(
            f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split: {split_mode}"
        )
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val` or `test`!"

        if os.path.exists(filename):
            print(
                f"INFO: negative samples for '{split_mode}' evaluation are already generated!"
            )
        else:
            print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
            # retrieve the information from the batch
            pos_src, pos_dst, pos_timestamp = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.t.cpu().numpy(),
            )

            # all possible destinations
            all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

            evaluation_set = {}
            # generate a list of negative destinations for each positive edge
            pos_edge_tqdm = tqdm(
                zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
            )
            for (
                pos_s,
                pos_d,
                pos_t,
            ) in pos_edge_tqdm:
                t_mask = pos_timestamp == pos_t
                src_mask = pos_src == pos_s
                fn_mask = np.logical_and(t_mask, src_mask)
                pos_e_dst_same_src = pos_dst[fn_mask]
                filtered_all_dst = np.setdiff1d(all_dst, pos_e_dst_same_src)

                '''
                when num_neg_e is larger than all possible destinations simple return all possible destinations
                '''

                if (self.num_neg_e is None or self.num_neg_e > len(filtered_all_dst)):
                    neg_d_arr = filtered_all_dst
                else:
                    neg_d_arr = np.random.choice(
                    filtered_all_dst, self.num_neg_e, replace=False) #never replace negatives

                evaluation_set[(pos_s, pos_d, pos_t)] = neg_d_arr

            # save the generated evaluation set to disk
            save_pkl(evaluation_set, filename)
