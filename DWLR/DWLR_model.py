import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from DWLR.backbone import F_Encoder, T_Encoder
from DWLR.common import DomainLoss, Classifier, Discriminator, ReweightNet, LabelAdapter
from DWLR.dataset import MyDataset

EPS = 1e-6


class DWLR(object):
    def __init__(self, config=None, save_path='./'):
        self.config = config
        save_path = os.path.join(save_path, f'model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model = DWLRNet(self.config, save_path)
        if self.config.cuda != 'cpu':
            self.model.cuda()

    def fit(self, x_source, y_source, x_target, n_epochs, learning_rate=None, y_target=None,
            x_test=None, y_test=None):
        source_dataset = MyDataset(x_source, y_source, self.config.n_freq)
        target_dataset = MyDataset(x_target, y_target, self.config.n_freq)
        test_dataset = None
        if x_test is not None:
            test_dataset = MyDataset(x_test, y_test, self.config.n_freq)

        if learning_rate is not None:
            self.config.lr = learning_rate

        self.model.set_src_class_count(source_dataset.class_count)
        pretrain_epoch = max((100, n_epochs//5))
        self.model.pretrain(source_dataset, pretrain_epoch)
        self.model.train_model(source_dataset, target_dataset, n_epochs, test_dataset)

    def predict(self, datas):
        dataset = MyDataset(datas)
        data_loader = DataLoader(dataset, dataset=256, shuffle=False, num_workers=16)
        prediction = []
        scores = []
        with torch.no_grad():
            for x, xa, xp, _ in data_loader:
                if self.config.cuda != 'cpu':
                    x, xa, xp = x.cuda(), xa.cuda(), xp.cuda()
                cls_out = self.model(x, xa, xp)
                score = torch.softmax(cls_out, dim=-1)
                _, pred = torch.max(score, dim=-1)

                prediction.append(pred.cpu().numpy())
                scores.append(score.cpu().numpy())
        
        prediction = np.concatenate(prediction)
        scores = np.concatenate(scores)
        return prediction, scores

    def load_model(self, model_path, idx=0):
        self.model.load(model_path, idx)


class DWLRNet(nn.Module):
    def __init__(self, config, save_path='./'):
        super(DWLRNet, self).__init__()
        self.config = config

        self.adv_loss_weight = config.adv_loss_weight
        self.reweight_loss_weight = config.reweight_weight

        self.f_encoder = F_Encoder(config)
        self.t_encoder = T_Encoder(config)
        
        self.f_classifier, self.f_reweight_net, self.f_discriminator = None, None, None
        if config.freq:
            self.f_classifier = Classifier(config.emb_dim, class_num=config.class_num, dp=0.3)
            self.f_reweight_net = ReweightNet(config.emb_dim)
            self.f_discriminator = Discriminator(config.emb_dim)
            
        self.t_classifier, self.t_reweight_net, self.t_discriminator = None, None, None
        if config.time:
            self.t_classifier = Classifier(config.emb_dim, class_num=config.class_num, dp=0.3)
            self.t_reweight_net = ReweightNet(config.emb_dim)
            self.t_discriminator = Discriminator(config.emb_dim)

        self.optimizer, self.reweight_optimizer, self.dis_optimizer = self.get_optim(config.lr, config.l2)

        self.label_adapter = LabelAdapter(config.class_num, config.cuda)
        self.domain_loss = DomainLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.domain_label0 = torch.zeros(config.batch_size, dtype=torch.long, requires_grad=False)
        self.domain_label1 = torch.ones(config.batch_size, dtype=torch.long, requires_grad=False)

        if config.cuda != 'cpu':
            self.domain_label0 = self.domain_label0.cuda()
            self.domain_label1 = self.domain_label1.cuda()
            self.cuda()

        self.save_path = save_path
        self.f_weights_obtain = []
        self.t_weights_obtain = []
        self.f_pseudo_lb = []
        self.t_pseudo_lb = []
    
    def get_optim(self, lr, l2):
        main_param_list = []
        reweight_param_list = []
        dis_param_list = []
        if self.config.freq:
            main_param_list.append({'params':self.f_encoder.parameters(),
                               'lr': lr, 'weight_decay': self.config.l2, 'betas':[0.9, 0.95], 'eps': EPS})
            main_param_list.append({'params': self.f_classifier.parameters()})
            reweight_param_list.append({'params': self.f_reweight_net.parameters()})
            dis_param_list.append({'params': self.f_discriminator.parameters()})
        if self.config.time:
            main_param_list.append({'params':self.t_encoder.parameters(),
                               'lr': lr, 'weight_decay': self.config.l2, 'betas':[0.9, 0.95], 'eps': EPS})
            main_param_list.append({'params': self.t_classifier.parameters()})
            reweight_param_list.append({'params': self.t_reweight_net.parameters()})
            dis_param_list.append({'params': self.t_discriminator.parameters()})
        
        main_optim = AdamW(main_param_list, lr=lr, weight_decay=l2)
        reweight_optim = AdamW(reweight_param_list, lr=lr, weight_decay=l2)
        dis_optim = AdamW(dis_param_list, lr=lr, weight_decay=l2)
        return main_optim, reweight_optim, dis_optim

    def set_src_class_count(self, class_count=None):
        if class_count is None:
            class_count = torch.ones(self.config.class_num)
        ce_weight = torch.tensor([max(class_count) / max((c, 1)) for c in class_count])

        if self.config.cuda != 'cpu':
            class_count = class_count.cuda()
            ce_weight = ce_weight.cuda()

        self.src_class = class_count
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=ce_weight)

    def get_f_weight(self, f_emb, cur_iter):
        return self.__get_weight(f_emb, cur_iter, self.f_classifier, self.f_reweight_net)

    def get_t_weight(self, t_emb, cur_iter):
        return self.__get_weight(t_emb, cur_iter, self.t_classifier, self.t_reweight_net)

    def __get_weight(self, emb, cur_iter, classifier, reweight_net):
        cls_out = torch.softmax(classifier(emb), -1)
        confidence, pseudo_labels = torch.max(cls_out.detach(), -1)

        # reweighting
        weights = reweight_net(emb)

        label_loss = self.label_adapter(weights, self.src_class, pseudo_labels)
        confidence_loss = -torch.mul(weights, confidence).mean()
        ig_loss = -torch.mul(weights, self.get_IG(emb.detach(), self.config.T, classifier)).mean()

        sample_loss = min((1000, cur_iter)) / 500 * label_loss * self.config.label_regular \
                 + confidence_loss * self.config.confidence \
                 + min((1000, cur_iter)) / 500 * ig_loss * self.config.IG

        return weights, pseudo_labels.detach(), sample_loss
    
    def get_IG(self, emb, T, classifier):
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        probs = []
        classifier.apply(apply_dropout)
        with torch.no_grad():
            for t in range(T):
                cls_out = classifier(emb)
                probs.append(torch.softmax(cls_out, dim=-1))
            probs = torch.stack(probs, dim=-1)
            _probs = torch.sum(probs, dim=-1) / T
            p_n = -torch.sum(torch.mul(_probs, torch.log(_probs + EPS)), dim=-1)
            p_t = -torch.sum(torch.sum(torch.mul(probs, torch.log(probs + EPS)), dim=-1) / T, dim=-1)

        return (p_n - p_t).detach()     

    def pretrain(self, dataset, epochs=10):
        for epoch in range(epochs):
            correct, total = 0, 0
            data_loader = DataLoader(dataset, self.config.batch_size, shuffle=True)
            for x, xa, xp, y in data_loader:
                if self.config.cuda != 'cpu':
                    x, xa, xp, y = x.cuda(), xa.cuda(), xp.cuda(), y.cuda()
                t_emb = self.t_encoder(x)
                f_emb = self.f_encoder(xa, xp)

                loss = 0.
                total_pred = None
                if t_emb is not None:
                    t_pred = self.t_classifier(t_emb)
                    loss += self.ce_loss(t_pred, y)
                    total_pred = t_pred
                if f_emb is not None:
                    f_pred = self.f_classifier(f_emb)
                    loss += self.ce_loss(f_pred, y)
                    total_pred = f_pred if total_pred is None else total_pred + f_pred

                self.optimizer.zero_grad()
                loss.backward()
                self.grad_norm()
                self.optimizer.step()

                _, pred = torch.max(total_pred, -1)
                correct += (pred == y).sum().item()
                total += pred.shape[0]
            acc = correct/total*100
            print('epoch:', epoch, 'acc:', acc)

    def train_step(self, x_s, xa_s, xp_s, y_s, x_t, xa_t, xp_t, cur_iter):
        if self.config.freq:
            f_emb_s = self.f_encoder(xa_s, xp_s)
            f_emb_t = self.f_encoder(xa_t, xp_t)
        if self.config.time:
            t_emb_s = self.t_encoder(x_s)
            t_emb_t = self.t_encoder(x_t)
        
        total_loss = 0.
        if self.config.time:
            t_weight, pseudo_label, t_sample_loss = self.get_t_weight(t_emb_t, cur_iter)
            self.t_weights_obtain.append(t_weight.detach().cpu())
            self.t_pseudo_lb.append(pseudo_label.detach().cpu())
            total_loss += t_sample_loss*self.reweight_loss_weight
            total_loss += self.ce_loss(self.t_classifier(t_emb_s), y_s)
            total_loss += self.adv_loss_weight*(self.domain_loss(self.t_discriminator(t_emb_s), self.domain_label0) +\
                        self.domain_loss(self.t_discriminator(t_emb_t), self.domain_label1, t_weight))
        if self.config.freq:
            f_weight, pseudo_label, f_sample_loss = self.get_f_weight(f_emb_t, cur_iter)
            self.f_weights_obtain.append(f_weight.detach().cpu())
            self.f_pseudo_lb.append(pseudo_label.detach().cpu())
            total_loss += f_sample_loss*self.reweight_loss_weight
            total_loss += self.ce_loss(self.f_classifier(f_emb_s), y_s)
            total_loss += self.adv_loss_weight*(self.domain_loss(self.f_discriminator(f_emb_s), self.domain_label0) +\
                        self.domain_loss(self.f_discriminator(f_emb_t), self.domain_label1, f_weight))

        self.optimizer.zero_grad()
        self.reweight_optimizer.zero_grad()
        total_loss.backward()
        self.grad_norm()
        self.reweight_optimizer.step()
        self.optimizer.step()

        # discriminator
        dis_loss = 0.
        if self.config.time:
            dis_loss += self.domain_loss(self.t_discriminator(t_emb_s.detach()), self.domain_label1) +\
                        self.domain_loss(self.t_discriminator(t_emb_t.detach()), self.domain_label0, t_weight.detach())
        if self.config.freq:
            dis_loss += self.domain_loss(self.f_discriminator(f_emb_s.detach()), self.domain_label1) +\
                        self.domain_loss(self.f_discriminator(f_emb_t.detach()), self.domain_label0, f_weight.detach())

        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()

    def grad_norm(self):
        if self.config.freq:
            nn.utils.clip_grad_norm_(self.f_encoder.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.f_classifier.parameters(), max_norm=0.5)
        if self.config.time:
            nn.utils.clip_grad_norm_(self.t_encoder.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.t_classifier.parameters(), max_norm=0.5)
        if self.config.inter:
            nn.utils.clip_grad_norm_(self.inter_t.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.inter_f.parameters(), max_norm=0.5)
    
    def train_model(self, source_dataset, target_dataset, epochs, test_dataset=None):
        for epoch in range(epochs):
            self.train()
            
            target_loader = DataLoader(target_dataset, batch_size=self.config.batch_size, shuffle=True)
            source_loader = DataLoader(source_dataset, batch_size=self.config.batch_size, shuffle=True)
            
            epoch_iter = max((len(target_loader), len(source_loader)))
            src_iter = iter(source_loader)
            tgt_iter = iter(target_loader)
            for i in range(epoch_iter):
                try:
                    x_s, xa_s, xp_s, y_s = next(src_iter)
                    if x_s.shape[0] != self.config.batch_size:
                        raise StopIteration()
                except StopIteration:
                    source_loader = DataLoader(source_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=16)
                    src_iter = iter(source_loader)
                    x_s, xa_s, xp_s, y_s = next(src_iter)

                try:
                    x_t, xa_t, xp_t, _ = next(tgt_iter)
                    if x_t.shape[0] != self.config.batch_size:
                        raise StopIteration()
                except StopIteration:
                    source_loader = DataLoader(target_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=16)
                    tgt_iter = iter(target_loader)
                    x_t, xa_t, xp_t, _ = next(tgt_iter)
                    
                if self.config.cuda != 'cpu':
                    x_s, xa_s, xp_s, y_s = x_s.cuda(), xa_s.cuda(), xp_s.cuda(), y_s.cuda()

                    x_t, xa_t, xp_t = x_t.cuda(), xa_t.cuda(), xp_t.cuda()
                
                self.train_step(x_s, xa_s, xp_s, y_s, x_t, xa_t, xp_t, epoch*epoch_iter+i)   
            train_acc, train_auc = self.evaluation(source_dataset)
            print('epoch', epoch)
            print('train acc:', train_acc, 'train_auc', train_auc)

        if test_dataset is not None:
            return self.evaluation(test_dataset)

    def evaluation(self, dataset):
        target_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)
        correct = 0
        total = 0
        scores = []
        labels = []
        self.eval()
        with torch.no_grad():
            for x, xa, xp, y in target_loader:
                if self.config.cuda != 'cpu':
                    x, xa, xp, y = x.cuda(), xa.cuda(), xp.cuda(), y.cuda()
                
                t_emb = self.t_encoder(x)
                f_emb = self.f_encoder(xa, xp)

                total_pred = None
                if t_emb is not None:
                    t_pred = self.t_classifier(t_emb)
                    total_pred = t_pred
                if f_emb is not None:
                    f_pred = self.f_classifier(f_emb)
                    total_pred = f_pred if total_pred is None else total_pred + f_pred

                score = torch.softmax(total_pred, dim=-1)
                _, pred = torch.max(score, dim=-1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
                scores.append(score.cpu().numpy())
                labels.append(y.cpu().numpy())
        
        acc = correct / total
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
        
        _eye = np.eye(self.config.class_num)
        marco_auc = roc_auc_score(_eye[labels], scores, average='macro') * 100
        return acc, marco_auc
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
    
    def load(self, save_path):
        model_path = os.path.join(save_path)
        self.load_state_dict(torch.load(model_path))
