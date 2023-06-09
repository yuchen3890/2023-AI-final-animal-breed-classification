import train_dataset # train
import torch
import os
import numpy as np
from io_utils import parse_args_eposide_train
import ResNet10
import ProtoNet
import torch.nn as nn
from torch.autograd import Variable
import utils
import random
import copy
import warnings
warnings.filterwarnings("ignore", category=Warning)
import  matplotlib.pyplot as plt
# from torch.nn.functional import pairwise_distance
# from torch.nn import TripletMarginLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params):
    model.train()
    top1 = utils.AverageMeter()
    total_loss = 0
    cls_loss = 0
    self_loss = 0
    cross_loss = 0
    local_loss = 0
    triplet_loss = 0
    eps = 1e-7
    softmax = torch.nn.Softmax(dim=1)

    # Initialize TripletMarginLoss function
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    for i, x in enumerate(train_loader):
        optimizer.zero_grad() 
        x_96 = torch.stack(x[2:8]).cuda() 
        x_224 = torch.stack(x[8:]).cuda() 
        support_set_anchor = x_224[0,:,:params.n_support,:,:,:] 
        query_set_anchor = x_224[0,:,params.n_support:,:,:,:] 
        query_set_aug_96 = x_96[:,:,params.n_support:,:,:,:]
        temp_224 = torch.cat((support_set_anchor, query_set_anchor), 1)
        temp_224 = temp_224.contiguous().view(params.n_way*(params.n_support+params.n_query),3,224,224) 
        temp_224 = model(temp_224) 
        temp_224 = temp_224.view(params.n_way, params.n_support+params.n_query, 512)
        support_set_anchor = temp_224[:,:params.n_support,:] 
        support_set_anchor = torch.mean(support_set_anchor, 1) 
        query_set_anchor = temp_224[:,params.n_support:,:] 
        query_set_anchor = query_set_anchor.contiguous().view(params.n_way*params.n_query, 512).unsqueeze(0)
        query_set_aug_96 = query_set_aug_96.contiguous().view(6*params.n_way*params.n_query,3,96,96)
        
        with torch.no_grad():
            query_set_aug_96 = Siamese_model(query_set_aug_96)
        query_set_aug_96 = query_set_aug_96.view(6, params.n_way*params.n_query, 512) 
        query_set = torch.cat((query_set_anchor, query_set_aug_96), 0)
        query_set = query_set.contiguous().view(7*params.n_way*params.n_query, 512)
        pred_query_set = head(support_set_anchor, query_set) 
        pred_query_set = pred_query_set.contiguous().view(7, params.n_way*params.n_query, params.n_way)
        pred_query_set_anchor = pred_query_set[0] 
        pred_query_set_aug = pred_query_set[1:] 

        query_set_y = torch.from_numpy(np.repeat(range(params.n_way), params.n_query))
        query_set_y = Variable(query_set_y.cuda())
        ce_loss = loss_fn(pred_query_set_anchor, query_set_y) 

        pred_query_set_anchor = softmax(pred_query_set_anchor)
        pred_query_set_aug = pred_query_set_aug.contiguous().view(6*params.n_way*params.n_query, params.n_way)
        pred_query_set_aug = softmax(pred_query_set_aug)
        pred_query_set_anchor = torch.cat([pred_query_set_anchor for _ in range(6)], dim=0)
        self_image_loss = torch.mean(torch.sum(torch.log((pred_query_set_aug + eps)**(-pred_query_set_anchor)), dim=1))

        pred_query_set_global = pred_query_set[0] 
        pred_query_set_global = pred_query_set_global.view(params.n_way, params.n_query, params.n_way)
        rand_id_global = np.random.permutation(params.n_query)
        pred_query_set_global = pred_query_set_global[:, rand_id_global[0], :] 
        pred_query_set_global = softmax(pred_query_set_global) 
        pred_query_set_global = pred_query_set_global.unsqueeze(0) 
        pred_query_set_global = pred_query_set_global.expand(6, params.n_way, params.n_way) 
        pred_query_set_global = pred_query_set_global.contiguous().view(6*params.n_way, params.n_way) 
        rand_id_local_sample = np.random.permutation(params.n_query)
        pred_query_set_local = pred_query_set_aug.view(6, params.n_way, params.n_query, params.n_way)
        pred_query_set_local = pred_query_set_local[:, :, rand_id_local_sample[0], :] 
        pred_query_set_local = pred_query_set_local.contiguous().view(6*params.n_way, params.n_way) 
        cross_image_loss = torch.mean(torch.sum(torch.log((pred_query_set_local + eps)**(-pred_query_set_global)), dim=1))

        # Compute anchor embeddings for triplet loss
        anchor_embeddings = []
        positive_embeddings = []
        negative_embeddings = []
        for class_idx in range(params.n_way):
            # get the images for the current class
            current_class_images = x_224[0, class_idx, :params.n_support, :, :]
            current_class_images_1 = x_224[0, class_idx, :5, :, :]
            current_class_embeddings = model(current_class_images.view(-1, 3, 224, 224))
            current_class_embeddings_1 = model(current_class_images_1.view(-1, 3, 224, 224))

            # randomly permute the indices and take the first two as anchor and positive
            permuted_indices = torch.randperm(current_class_embeddings.shape[0])
            permuted_indices_1 = torch.randperm(current_class_embeddings_1.shape[0])
            anchor_idx = permuted_indices[0]
            positive_idx = permuted_indices_1[1]
            anchor_embeddings.append(current_class_embeddings[anchor_idx])
            positive_embeddings.append(current_class_embeddings_1[positive_idx])

            # for the negative embedding, pick one sample from a different class
            negative_class_idx = (class_idx + 1) % params.n_way  # this guarantees that we pick a different class
            negative_class_images = x_224[0, negative_class_idx, :params.n_support, :, :]
            negative_class_embeddings = model(negative_class_images.view(-1, 3, 224, 224))
            negative_idx = torch.randint(0, negative_class_embeddings.shape[0], (1,))
            negative_embeddings.append(negative_class_embeddings[negative_idx].squeeze(0))

 
        # Convert lists to tensors
        anchor_embeddings = torch.stack(anchor_embeddings).cuda()
        positive_embeddings = torch.stack(positive_embeddings).cuda()
        negative_embeddings = torch.stack(negative_embeddings).cuda()
        # print(anchor_embeddings.shape)
        # print(positive_embeddings.shape)
        # print(negative_embeddings.shape)

        # Calculate the triplet loss
        triplet_loss = triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Add the triplet loss to the total loss
        loss = ce_loss + self_image_loss * params.lamba1 + cross_image_loss * params.lamba2 + triplet_loss * params.lamba3

        _, predicted = torch.max(pred_query_set[0].data, 1)
        correct = predicted.eq(query_set_y.data).cpu().sum()
        top1.update(correct.item()*100 / (query_set_y.size(0)+0.0), query_set_y.size(0))  
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param_q, param_k in zip(model.parameters(), Siamese_model.parameters()):
                param_k.data = param_k.data * params.m + param_q.data * (1. - params.m)
    
        total_loss = total_loss + loss.item()
        cls_loss = cls_loss + ce_loss.item()
        self_loss = self_loss + self_image_loss.item()
        cross_loss = cross_loss + cross_image_loss.item()
        triplet_loss_value = triplet_loss.item()
    avg_loss = total_loss/float(i+1)
    avg_triplet_loss = triplet_loss_value/float(i+1)
    avg_ce_loss = cls_loss/float(i+1)
    avg_self_image_loss = self_loss/float(i+1)
    avg_cross_image_loss = cross_loss/float(i+1)
    return avg_loss, top1.avg, avg_triplet_loss, avg_ce_loss, avg_self_image_loss, avg_cross_image_loss

 
                
if __name__=='__main__':

    params = parse_args_eposide_train()

    setup_seed(params.seed)

    datamgr_train = train_dataset.Eposide_DataManager(data_path=params.source_data_path, num_class=params.train_num_class, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.train_n_eposide)
    train_loader = datamgr_train.get_data_loader()

    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)

    head = ProtoNet.ProtoNet()

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state)
    Siamese_model = copy.deepcopy(model)
    model = model.cuda()
    Siamese_model = Siamese_model.cuda()
    head = head.cuda()

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params":model.parameters()}], lr=params.lr)
    best_acc = 0
    train_loss_all = []
    ce_loss_all = []
    self_image_loss_all = []
    cross_image_loss_all = []
    triplet_loss_all = []
    for epoch in range(params.epoch):
        # train_loss, train_acc, triplet_loss_1 = train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params)
        train_loss, train_acc, triplet_loss_1, ce_loss, self_image_loss, cross_image_loss = train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params)
        print('train:', epoch+1, 'current epoch train loss:', train_loss, 'current epoch train acc:', train_acc , 'current epoch train triplet_loss:', 100*triplet_loss_1)
        train_loss_all.append(train_loss)
        ce_loss_all.append(ce_loss)
        self_image_loss_all.append(self_image_loss)
        cross_image_loss_all.append(cross_image_loss)
        triplet_loss_all.append(100*triplet_loss_1)

        if train_acc > best_acc:
            best_acc = train_acc
            outfile = os.path.join(params.save_dir, 'best.tar')
            torch.save({
            'epoch':epoch+1,
            'state_model':model.state_dict(),
            'state_Siamese_model':Siamese_model.state_dict()},
            outfile)

    plt.plot(train_loss_all, label='train_loss')
    plt.plot(ce_loss_all, label='cls_loss')
    plt.plot(self_image_loss_all, label='self_loss')
    plt.plot(cross_image_loss_all, label='cross_loss')
    plt.plot(triplet_loss_all, label='triplet_loss')
    plt.legend(loc = 'upper right')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(params.save_dir, 'loss_curve.png'))
    
    outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch+1))
    torch.save({
    'epoch':epoch+1, 
    'state_model':model.state_dict(),
    'state_Siamese_model':Siamese_model.state_dict()},
     outfile) 


    
    
    
    