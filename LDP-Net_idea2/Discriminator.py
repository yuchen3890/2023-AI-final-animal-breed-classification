import torch.nn as nn
import torch

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, iter_num, alpha, low, high, max_iter):
        ctx.save_for_backward(iter_num, alpha, low, high, max_iter)
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        iter_num, alpha, low, high, max_iter = ctx.saved_tensors
        grad_output = grad_output.clone()
        coeff = 2.0 * (high - low) / (1.0 + torch.exp(-alpha * iter_num / max_iter))- (high - low) + low
        return -coeff * grad_output, None, None, None, None, None


def grad_reverse(x, iter_num, alpha, low, high, max_iter):
    iter_num = torch.tensor(iter_num)
    alpha = torch.tensor(alpha)
    low = torch.tensor(low)
    high = torch.tensor(high)
    max_iter = torch.tensor(max_iter)
    return GRL.apply(x, iter_num, alpha, low, high, max_iter)

class Discriminator(nn.Module):
    def __init__(self, dim=512, n_classes=3):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(dim, n_classes)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()
        self.grl_iter_num = 0
        self.grl_alpha = 10
        self.grl_low = 0.0
        self.grl_high = 1.0
        self.grl_max_iter = 4000

    def forward(self, feature):
        #adversarial_out = self.ad_net(self.grl_layer(feature))
        self.grl_iter_num += 1
        adversarial_out = self.ad_net(grad_reverse(feature, self.grl_iter_num, self.grl_alpha, self.grl_low, self.grl_high, self.grl_max_iter))
        return adversarial_out