import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


class DomainLoss(nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()
        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, sample_idx=None):
        loss_out = self.ce_loss_none(pred, target)
        if sample_idx is not None:
            loss = torch.mul(loss_out, sample_idx).mean()
        else:
            loss = loss_out.mean()
        return loss
    

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, EPS)



class Classifier(nn.Module):
    def __init__(self, feature_dim, class_num, dp=0.1):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dp)
        self.cls = nn.Linear(feature_dim, class_num)

    def forward(self, emb):
        cls_out = self.cls(self.dropout(emb))
        return cls_out


class Discriminator(nn.Module):
    def __init__(self, feature_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2, bias=False),
            nn.BatchNorm1d(feature_dim*2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(feature_dim*2, feature_dim*2, bias=False),
            nn.BatchNorm1d(feature_dim*2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(feature_dim*2, 2),
        )

    def forward(self, emb):
        return self.net(emb)


class ReweightNet(nn.Module):
    def __init__(self, feature_dim):
        super(ReweightNet, self).__init__()
        self.weightNet = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        raw_weight = self.weightNet(x).reshape(-1)  
        weights = F.gumbel_softmax(raw_weight, dim=0, hard=False) * x.shape[0]
        return weights


class LabelAdapter(nn.Module):
    def __init__(self, class_num=5, cuda='cpu', crit='kl'): 
        super(LabelAdapter, self).__init__()
        self.kl_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.cos = nn.CosineSimilarity(-1)
        self.a = torch.eye(class_num)
        if cuda != 'cpu':
            self.a = self.a.cuda()
        self.logsoftmax = nn.LogSoftmax(-1)
        self.crit = crit

    def forward(self, tgt_weight, src_class, tgt_labels):
        tgt_class = self.get_class_count(tgt_weight, tgt_labels)
        if self.crit == 'cos':
            return 1 - self.cos(tgt_class, src_class)
        tgt_class = self.logsoftmax(tgt_class*1.0/tgt_class.sum()).unsqueeze(0)
        src_class = self.logsoftmax(src_class*1.0/src_class.sum()).unsqueeze(0)
        return self.kl_loss(tgt_class, src_class)

    def get_class_count(self, select_count, labels):
        label_onehot = self.a[labels]
        class_count = torch.matmul(select_count, label_onehot)
        return class_count
