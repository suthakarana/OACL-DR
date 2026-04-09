import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoFixed(nn.Module):
    def __init__(self, num_classes: int, dim: int, K: int):
        super().__init__()

        self.num_classes = num_classes
        self.dim = dim
        self.K = K

        # (C, K, D)
        self.register_buffer("queue", torch.randn(num_classes, K, dim))
        # self.queue = F.normalize(self.queue, dim=2)

        # pointer per class
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor, labels: torch.Tensor):
        """
        keys: (B, D)
        labels: (B,)
        """
        # keys = F.normalize(keys, dim=1)

        for c in labels.unique():
            c = c.item()
            mask = labels == c
            class_keys = keys[mask]
            Nc = class_keys.size(0)

            ptr = int(self.queue_ptr[c])

            if Nc >= self.K:
                self.queue[c] = class_keys[-self.K:]
                self.queue_ptr[c] = 0
                continue

            end_ptr = ptr + Nc

            if end_ptr <= self.K:
                self.queue[c, ptr:end_ptr] = class_keys
            else:
                first_part = self.K - ptr
                self.queue[c, ptr:] = class_keys[:first_part]
                self.queue[c, :end_ptr - self.K] = class_keys[first_part:]

            self.queue_ptr[c] = (ptr + Nc) % self.K

    @torch.no_grad()
    def getDict(self):
        """
        Returns:
            features: (C*K, D)
            labels:   (C*K,)
        """
        C, K, D = self.queue.shape

        # reshape features
        features = self.queue.view(C * K, D)

        # create labels
        labels = torch.arange(C, device=self.queue.device)
        labels = labels.unsqueeze(1).expand(C, K).reshape(-1)

        return features, labels
