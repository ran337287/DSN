import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutil
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model import DSN
from test import test
import numpy as np
import functions as func

source_dataset_name = 'SVHN'
target_dataset_name = 'mnist'
source_dataset = os.path.join('.', 'dataset', 'svhn')
target_dataset = os.path.join('.', 'dataset', 'mnist')
model_root = 'models'   # directory to save trained models
cuda = True
cudnn.benchmark = True
lr = 1e-4
batch_size = 64
image_size = 32
n_channels = 3
n_epoch = 200
weight_decay = 1e-6
lr_decay_epoch = 30
decay_weight = 0.1

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data, gain=1)
        nn.init.constant(m.bias.data, 0.1)

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
def norm_minmax(img,min_norm=0.0,max_norm=1.0):
    '''
    input size [batch_size,c,h,w]
    output size [batch_size,c,h,w]
    '''
    avg_img = torch.mean(img,0)
    min_v = torch.min(avg_img)
    max_v = torch.max(avg_img)
    img[img > max_v] = max_v
    img[img < min_v] = min_v
    img = (img - min_v) / (max_v -min_v) * (max_norm - min_norm) + min_norm
    return img

def exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=lr_decay_epoch, decay_weight=0.1):
    lr = init_lr*(decay_weight**(epoch/lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


img_src_transform = transforms.Compose([
    #transforms.ColorJitter(0.4)
    # transforms.RandomResizedCrop(image_size),
    transforms.Resize(image_size),
    # transforms.Grayscale(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    # transforms.Lambda(norm_minmax),
    # transforms.Normalize(mean=(0.5,),std=(0.5,))
])

img_tgt_transform = transforms.Compose([
    transforms.Resize(image_size),
    # transforms.RandomResizedCrop(image_size),
    # transforms.Grayscale(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    # transforms.Lambda(norm_minmax),
    # transforms.Normalize(mean=(0.5,),std=(0.5,))
])

dataset_source = datasets.SVHN(
    root=source_dataset,
    split='train',
    transform=img_src_transform,
)

datasetloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

dataset_target = datasets.MNIST(
    root=target_dataset,
    train=True,
    transform=img_tgt_transform,
)

datasetloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

# load models
my_net = DSN(n_class=10,code_size=3072,channels=n_channels)
my_net.apply(weights_init)

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = optim.RMSprop(my_net.parameters(), lr=lr, weight_decay=weight_decay)

loss_class = nn.CrossEntropyLoss()
loss_rec = func.mean_pairwise_square_loss()
loss_diff = func.difference_loss()
if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_rec = loss_rec.cuda()
    loss_diff = loss_diff.cuda()

#loss coefficients
coeff_alpha = 0.07 * torch.ones(1)
coeff_beta = 0.07 * torch.ones(1)
coeff_gamma = 0.25 * torch.ones(1)
#train
Rec_scheme = 'all'#'private','shared','all'
if cuda:
    coeff_alpha = coeff_alpha.cuda()
    coeff_beta = coeff_beta.cuda()
    coeff_gamma = coeff_gamma.cuda()

coeff_alpha = Variable(coeff_alpha)
coeff_beta = Variable(coeff_beta)
coeff_gamma = Variable(coeff_gamma)

for p in my_net.parameters():
    p.requires_grad = True

len_source = len(datasetloader_source)
len_target = len(datasetloader_target)
len_iter = min(len_source,len_target)


dann_iter = 5000
global_iter = 0
for epoch in xrange(n_epoch):
    dataset_source_iter = iter(datasetloader_source)
    dataset_target_iter = iter(datasetloader_target)
    # optimizer = exp_lr_scheduler(optimizer, epoch, lr, lr_decay_epoch, decay_weight)
    i = 0
    while i < len_iter:
        my_net.zero_grad()
        p_alpha = 0.0
        if global_iter > dann_iter-1:
            p_alpha = 1.0
        # p = float(i + epoch * len_iter) / n_epoch / len_iter
        p = p_alpha * (global_iter - dann_iter) /(100 *len_iter - dann_iter )
        p = min(p,1.0)
        alpha = 2. / (1. + np.exp( -10 * p))-1
        ##### target data ###########
        data_target = dataset_target_iter.next()
        t_img, _ = data_target
        vutil.save_image(t_img, 't_gray.png')
        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, n_channels, image_size, image_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()
        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        inputv_img = Variable(input_img)
        domainv_label = Variable(domain_label)
        ############################
        pri_tgt_feat, shd_tgt_feat, _, pred_tgt_domain, img_tgt_rec = my_net(inputv_img, 'target', Rec_scheme, alpha)

        p_alpha = 0.0
        if global_iter > dann_iter - 1:
            p_alpha = 1.0

        err_t_domain = p_alpha * loss_class(pred_tgt_domain, domainv_label)
        # rec_loss
        # t_rec_img = norm_minmax(img_tgt_rec.data)
        # vutil.save_image(t_rec_img, 't_rec_img.png', nrow=8)
        vutil.save_image(img_tgt_rec.data, 't_rec_img.png', nrow=8)
        # img_tgt_rec = img_tgt_rec.view(-1, n_channels * image_size * image_size)

        t_ori_img = inputv_img.expand(inputv_img.data.shape[0], n_channels, image_size, image_size)
        # t_ori_img = t_ori_img.contiguous().view(-1, n_channels * image_size * image_size)
        err_t_rec = loss_rec(img_tgt_rec, t_ori_img)
        # diff_loss
        diff_t_loss = loss_diff(pri_tgt_feat, shd_tgt_feat)

        tgt_loss = coeff_alpha * err_t_rec \
                   + coeff_beta * diff_t_loss + coeff_gamma * err_t_domain

        tgt_loss.backward()
        optimizer.step()

        my_net.zero_grad()
        ##### source data ###########
        data_source = dataset_source_iter.next()
        s_img, s_label = data_source
        # ss_img = norm_minmax(s_img,0,1)

        # vutil.save_image(ss_img, 'ss_gray.png')
        vutil.save_image(s_img, 's_gray.png')
        s_label = s_label.long().squeeze()

        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, n_channels, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        ################
        pri_src_feat, shd_src_feat, pred_label, pred_src_domain, img_src_rec = my_net(inputv_img,'source',Rec_scheme,alpha)

        err_s_label = loss_class(pred_label, classv_label)

        p_alpha = 0.0
        if global_iter > dann_iter-1:
            p_alpha = 1.0

        err_s_domain = p_alpha * loss_class(pred_src_domain, domainv_label)

        #rec_loss
        # s_rec_img = norm_minmax(img_src_rec.data)
        # vutil.save_image(s_rec_img, 's_rec_img.png', nrow=8)
        vutil.save_image(img_src_rec.data, 's_rec_img.png', nrow=8)

        # img_src_rec = img_src_rec.view(-1, n_channels * image_size * image_size)
        s_ori_img = inputv_img
        # s_ori_img = inputv_img.contiguous().view(-1, n_channels * image_size * image_size)
        err_s_rec = loss_rec(img_src_rec, s_ori_img)

        # diff_loss
        diff_s_loss = loss_diff(pri_src_feat, shd_src_feat)

        src_loss = err_s_label + coeff_alpha * err_s_rec \
                + coeff_beta * diff_s_loss + coeff_gamma * err_s_domain

        src_loss.backward()
        optimizer.step()



        ############ Loss  and upgrade gradient #################
        Loss_class = err_s_label
        Loss_similar = err_s_domain + err_t_domain
        Loss_diff = diff_s_loss + diff_t_loss
        diff_loss = diff_s_loss + diff_t_loss
        Loss_rec = err_s_rec + err_t_rec
        loss = Loss_class + coeff_alpha * Loss_rec \
                + coeff_beta * Loss_diff + coeff_gamma * Loss_similar


        if ((i % 100 == 0) | (i == (len_iter - 1))):
            print 'epoch: %d, [iter: %d / all %d], [err_s_label: %4f]' \
                  % (epoch, i, len_iter, err_s_label.cpu().data.numpy())
            print '[err_s_domain: %4f / err_t_domain %4f], [diff_s_loss %4f/ diff_t_loss %4f]' \
                  % (err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy(),
                     diff_s_loss.cpu().data.numpy(), diff_t_loss.cpu().data.numpy())
            print '[err_s_rec: %4f/ err_t_rec %4f], loss %4f' \
                  % (err_s_rec.cpu().data.numpy(), err_t_rec.cpu().data.numpy(), loss.cpu().data.numpy())
            print '--------------------------------------------------------'

        i += 1
        global_iter += 1

    torch.save(my_net, '{0}/svhn_mnist_model_epoch_{1}.pth'.format(model_root, epoch))
    test(epoch)

print 'done'




