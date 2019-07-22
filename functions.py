from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx,x,alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

import torch
import torch.nn as nn
import torch.nn.functional as func

class difference_loss(nn.Module):
    def __init__(self,reduce=True):
        super(difference_loss,self).__init__()
        self.reduce = reduce
        return

    def forward(self, private_samples,shared_samples):
        batch_size = private_samples.size(0)
        private_samples = private_samples.view(batch_size,-1)
        shared_samples = shared_samples.view(batch_size,-1)

        private_samples = private_samples - torch.mean(private_samples,0,keepdim=True)
        shared_samples = shared_samples - torch.mean(shared_samples,0,keepdim=True)

        pn = torch.norm(private_samples,p=2,dim=1, keepdim=True)
        sn = torch.norm(shared_samples,p=2,dim=1, keepdim=True)

        private_samples = private_samples.div(pn.expand_as(private_samples)+1e-10)
        shared_samples = shared_samples.div(sn.expand_as(shared_samples)+1e-10)

        diff_loss = torch.sum((shared_samples.t().mm(private_samples)).pow(2), 0)

        if self.reduce:
            diff_loss = torch.mean(diff_loss)

        return diff_loss

class mean_pairwise_square_loss(nn.Module):
    def __init__(self,reduce=True):
        super(mean_pairwise_square_loss,self).__init__()
        self.reduce = reduce
        return

    def forward(self,predictions,labels):
        diff = predictions - labels
        diff = diff.view(diff.size(0),-1)

        sum_square_diff = torch.sum(diff.pow(2))
        square_sum_diff = torch.sum(torch.pow(torch.sum(diff,1),2))

        num_present =torch.numel(diff.data)
        loss = sum_square_diff / num_present + square_sum_diff / num_present / num_present

        if not self.reduce:
            loss = loss * diff.size(0)

        return loss
