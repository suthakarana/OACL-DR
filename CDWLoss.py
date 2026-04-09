import torch
import torch.nn as nn
import torch.nn.functional as F

class CDWloss(nn.Module):
    def __init__(self, nclass, gamma_ordinal,gpuid, lambda_mse ,temperature=0.1, w_neg = 1,
                 anneal_epochs=30, mode="linear"):
        super().__init__()

        self.temperature = temperature
        self.gamma_ordinal = gamma_ordinal
        self.anneal_epochs = anneal_epochs
        self.mode = mode
        self.w_neg = w_neg
        self.nclass = nclass
        self.lambda_mse = lambda_mse
        self.gpuid =gpuid

    def get_class_weights(self, epoch):
        finalVal = 1.0/self.nclass
        if epoch >= self.anneal_epochs:
            return torch.ones_like(self.w0)*finalVal
        t = epoch / self.anneal_epochs
        if self.mode == "linear":
            s = t
        elif self.mode == "cosine":
            s = 0.5 * (1 - torch.cos(torch.tensor(t * 3.1415926535, device=self.w0.device)))
        else:
            raise ValueError("mode must be 'linear' or 'cosine'")
        return (1 - s) * self.w0 + s * finalVal


    def forward(self, x, y, x_moco, y_moco,   prototypes=None):
        logits = self.__getSim(x, x_moco)

        distance = (y.unsqueeze(1).float() - y_moco.unsqueeze(0).float()).abs()

        m = 1.0 + self.w_neg * (distance ** self.gamma_ordinal)

        adjusted_logits = logits + torch.log(m)
        log_prob = F.log_softmax(adjusted_logits, dim=1)

        pos_mask = (distance == 0).float()

        pos_count = pos_mask.sum(dim=1) 
        has_pos = pos_count > 0

        pos_logprob_sum = -(pos_mask * log_prob).sum(dim=1)
        per_sample_loss = pos_logprob_sum / pos_count.clamp(min=1.0)
        per_sample_loss = per_sample_loss * has_pos.float()
        order_loss = per_sample_loss.mean()

        return order_loss

    def getProbClassInference(self, fea, prototypes):
        sim = self.__getSim(fea, prototypes)
        return F.softmax(sim, dim=1)


    def __getSim(self, x, x_moco):
        x = F.normalize(x, dim=1)
        x_moco = F.normalize(x_moco, dim=1)
        sim = x.mm(x_moco.T)
        sim = torch.div(sim, self.temperature)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        return sim

    def getProbClass(self, x, x_moco, y_moco):
        exp_sim = torch.exp(self.__getSim(x, x_moco))

        bs = x.shape[0]
        y_moco = y_moco.view(-1, 1)

        sim_class = torch.zeros(bs, self.nclass).cuda(self.gpuid)
        for c in range(self.nclass):
            idx = (y_moco == c).view(-1)
            if idx.sum() > 0:
                sim_class[:, c] = exp_sim[:, idx].sum(1) / idx.sum()
        prob = sim_class / torch.sum(sim_class, keepdim=True, dim=1).clamp(1e-10)
        return prob
