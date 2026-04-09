import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

class MoCo(nn.Module):

    def __init__(self, gpuid, nclasses, ndim, K):

        super(MoCo, self).__init__()
        self.K = K
        self.ndim = ndim
        self.nclasses = nclasses

        # create the queue
        self.gpuid = gpuid
        self.init_buffers()

    def init_buffers(self):
        self.register_buffer("queue", torch.randn(self.K, self.ndim))
        self.register_buffer("queue_y", torch.ones(self.K, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = self.queue.cuda(self.gpuid)
        self.queue_y = self.queue_y.cuda(self.gpuid)
        self.queue_ptr = self.queue_ptr.cuda(self.gpuid)


    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, y):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size]
            y = y[:batch_size]

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr: ptr + batch_size, :] = keys
        self.queue_y[ptr: ptr + batch_size] = y
        ptr = (ptr + batch_size) % self.K   
        self.queue_ptr[0] = ptr

    def getDict(self):
        idx = self.queue_y != -1
        if idx.sum() > 0:
            return self.queue[idx].clone().detach(), self.queue_y[idx]
        return None, None


