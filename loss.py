import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
import numpy as np

from lorentz import LorentzCalculation
              



class HyperbolicCQCLoss(nn.Module):

    def __init__(self, tau_cqc, writer=None, assymetric_mode=False):
        super(HyperbolicCQCLoss, self).__init__()
        self.tau_cqc = tau_cqc
        self.lorentz_calculator = LorentzCalculation()
        self.CE = nn.CrossEntropyLoss(reduction="mean")
        self.writer = writer
        self.global_step = 0
        self.assymetric_mode = assymetric_mode
     

    def _get_correlated_mask(self, batch_size, device):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(device)

    def compute_lorentz_similarity_matrix(self, X, neg_curvs):
        """
        Compute Lorentz similarity matrix for hyperbolic embeddings.
        X: [N, M, D] where N is batch size, M is number of components, D is embedding dim
        Returns: [N, N] similarity matrix
        """
        N, M, D = X.shape
        dist_components = []
        for i in range(M):
            # Compute pairwise Lorentz distances for component i
            dist_i = self.lorentz_calculator.lorentz_dist(
                X[:, i, :],  # [N, D]
                X[:, i, :],  # [N, D] - same tensor for pairwise distances
                neg_curvs[i]
            )  # [N, N]
            dist_components.append(dist_i)
            
        total_dist = torch.stack(dist_components, dim=-1).sum(dim=-1)  # [N, N]
            
        similarity = -total_dist
        
        return similarity
    
   
    
    def assymetric_forward(self, Xa, Xb, Za, Zb, neg_curvs):
        self.global_step += 1
        B = Za.shape[0]
        device = Za.device

        sim_XaZb = self.compute_lorentz_similarity_matrix(
            torch.cat([Xa, Zb], dim=0), neg_curvs
        )[:B, B:]  # [B,B]
        sim_XbZa = self.compute_lorentz_similarity_matrix(
            torch.cat([Xb, Za], dim=0), neg_curvs
        )[:B, B:]  

        logits_XaZb = sim_XaZb / self.tau_cqc  
        logits_XbZa = sim_XbZa / self.tau_cqc 

        labels = torch.arange(B, device=device).long()
        loss = 0.5 * (self.CE(logits_XaZb, labels) + self.CE(logits_XbZa, labels))

        if self.writer is not None:
            self.writer.add_scalar('cqc/assym_loss', loss.item(), self.global_step)
        return loss

    def forward(self, Xa, Xb, Za, Zb, neg_curvs, im2cluster=None):
        if self.assymetric_mode:
            return self.assymetric_forward(Xa, Xb, Za, Zb, neg_curvs)
        self.global_step += 1
        batch_size = Xa.shape[0]
        device = Xa.device

    
        XaZb = torch.cat([Xa, Zb], dim=0)  # [2*batch_size, M, D]
        XbZa = torch.cat([Xb, Za], dim=0)  # [2*batch_size, M, D]

        # Compute Lorentz similarity matrices
        sim_ab = self.compute_lorentz_similarity_matrix(XaZb, neg_curvs)  # [2*batch_size, 2*batch_size]
        sim_ba = self.compute_lorentz_similarity_matrix(XbZa, neg_curvs)  # [2*batch_size, 2*batch_size]

        # Get correlation mask
        get_corr_mask = self._get_correlated_mask(batch_size, device)

        # Extract positive pairs (diagonal elements)
        Rab = torch.diag(sim_ab, batch_size)
        Lab = torch.diag(sim_ab, -batch_size)
        Pos_ab = torch.cat([Rab, Lab]).view(2 * batch_size, 1)
        Neg_ab = sim_ab[get_corr_mask].view(2 * batch_size, -1)

        Rba = torch.diag(sim_ba, batch_size)
        Lba = torch.diag(sim_ba, -batch_size)    
        Pos_ba = torch.cat([Rba, Lba]).view(2 * batch_size, 1)
        Neg_ba = sim_ba[get_corr_mask].view(2 * batch_size, -1)
           
        # Create logits
        logits_ab = torch.cat((Pos_ab, Neg_ab), dim=1)
        logits_ab /= self.tau_cqc

        logits_ba = torch.cat((Pos_ba, Neg_ba), dim=1)
        logits_ba /= self.tau_cqc

     
        
        labels = torch.zeros(2 * batch_size, device=device).long()
        cqc_loss = 0.5 * (self.CE(logits_ab, labels) + self.CE(logits_ba, labels))
        
        
        neighbor_inst_loss = torch.tensor(0.0, device=device)
        if im2cluster is not None:
            nbr_losses = []

            sim_i_ab = sim_ab[:batch_size, :batch_size]
            sim_i_ba = sim_ba[:batch_size, :batch_size]

            def _neighbor_mask(im2c: torch.Tensor) -> torch.Tensor:
                im2c = im2c.view(-1, 1)
                mat = (im2c == im2c.T).float()
                mat.fill_diagonal_(0.)
                idx = torch.argmax(mat, dim=1, keepdim=True)
                sel = torch.zeros_like(mat, device=device)
                sel.scatter_(1, idx, 1.0)
                sel = sel * mat
                return sel.bool()

            for ith in im2cluster:
                nb = _neighbor_mask(ith.to(device))
                rows = nb.any(dim=1)
                if rows.any():
                    # Get the filtered neighbor mask and create proper labels
                    nb_filtered = nb[rows][:, rows]
                    y = torch.argmax(nb_filtered.float(), dim=1)
                    
                    # Ensure labels are within valid range
                    y = torch.clamp(y, 0, nb_filtered.shape[1] - 1)

                    logits_n_ab = sim_i_ab[rows][:, rows] / self.tau_cqc
                    nbr_losses.append(self.CE(logits_n_ab, y))

                    logits_n_ba = sim_i_ba[rows][:, rows] / self.tau_cqc
                    nbr_losses.append(self.CE(logits_n_ba, y))

            if nbr_losses:
                neighbor_inst_loss = sum(nbr_losses) / len(nbr_losses)

        if self.writer is not None:
            self.writer.add_scalar('cqc/loss', float(cqc_loss.item()) if isinstance(cqc_loss, torch.Tensor) else float(cqc_loss), self.global_step)
            if im2cluster is not None:
                nb_val = float(neighbor_inst_loss.item()) if isinstance(neighbor_inst_loss, torch.Tensor) else float(neighbor_inst_loss)
                self.writer.add_scalar('cqc/neighbor_loss', nb_val, self.global_step)

        return cqc_loss, neighbor_inst_loss


class ProtoLoss(nn.Module):

    def __init__(self, temp):
        super(ProtoLoss, self).__init__()
        self.temp = temp
        self.lorentz_calculator = LorentzCalculation()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, view1_hat_feats, neg_curvs, tangent_to_hyper_func, clus_mode, cluster_result=None, index=None):
        """
        Input:
            view1_hat_feats: quantized embeddings of a batch of query images, [b, M, D]
            cluster_result: cluster assignments, centroids, and density. Shape of centroids (in product of tangent space) is: [num_cluster, M * D]
            index: indices for training samples
        Output:
            proto_losses: list of prototypical losses that corresponds to each num_cluster
        """
        assert cluster_result is not None, "cluster_result must be provided for ProtoLoss.forward"
        proto_loss_list = []
        prev_pos_prototypes = None
        for step, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster'], 
                                                                     cluster_result['centroids'])):
            # get positive prototypes
            pos_proto_id = im2cluster[index]
            pos_prototypes = prototypes[pos_proto_id]

            # sample negative prototypes, 
            all_proto_id = [i for i in range(im2cluster.max()+1)]       
            neg_proto_id = list(set(all_proto_id)-set(pos_proto_id.tolist()))
            # neg_proto_id = sample(neg_proto_id, self.r) #sample r negative prototypes. Do not sample
            neg_prototypes = prototypes[neg_proto_id]    

            # if clus_mode == "hier_residual" and step>0:
            #     proto_selected = pos_prototypes
            # else:
            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0) # [bsz+len(neg), M*D], in tangent space

            hyper_proto_selected = tangent_to_hyper_func(proto_selected) # [bsz+len(neg), M, D] in hyper space

            # compute lorentzian dist
            dist = [self.lorentz_calculator.lorentz_dist(view1_hat_feats[:,i,:], hyper_proto_selected[:,i,:], neg_curvs[i]) for i in range(view1_hat_feats.shape[1])] # [bsz, bsz+len(neg)]
            dist = torch.stack(dist, dim=-1) # [bsz  bsz+len(neg), M]
            dist = dist.sum(dim = -1) # [bsz, bsz+len(neg)]
            logits_proto = -dist # [bsz, bsz+len(neg)]

            # targets for prototype assignment
            labels_proto = torch.linspace(0, view1_hat_feats.size(0)-1, steps=view1_hat_feats.size(0)).long().cuda()

            # scaling temperatures for the selected prototypes
            # if clus_mode == "hier_residual" and step>0:
            #     temp_proto = density[pos_proto_id]
            # else:
            logits_proto /= self.temp 
                
            # logits_proto /= temp_proto


            cur_proto_loss = self.criterion(logits_proto, labels_proto) / view1_hat_feats.shape[0]
            # cur_proto_loss = -torch.diagonal(logits_proto).mean()

            proto_loss_list.append(cur_proto_loss)

        return proto_loss_list


