import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets

def rec_image(epoch,mode,Rec_scheme,Issource):

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 32
    n_channels = 3

    # load data
    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        # transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    if mode == 'mnist_trn':
        model_root = 'models'
        image_root = os.path.join('dataset', 'mnist')
        dataset = datasets.MNIST(
            root=image_root,
            train=True,
            transform=img_transform)
    if mode == 'mnist_tst':
        model_root = 'models'
        image_root = os.path.join('dataset', 'mnist')
        dataset = datasets.MNIST(
            root=image_root,
            train=False,
            transform=img_transform)

    if mode == 'svhn_trn':
        model_root = 'models'
        image_root = os.path.join('dataset', 'svhn')

        dataset = datasets.SVHN(
            root=image_root,
            split='train',
            transform=img_transform)
    if mode == 'svhn_tst':
        model_root = 'models'
        image_root = os.path.join('dataset', 'svhn')

        dataset = datasets.SVHN(
            root=image_root,
            split='test',
            transform=img_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    # test
    my_net = torch.load(os.path.join(
        model_root, 'svhn_mnist_model_epoch_' + str(epoch) + '.pth')
    )

    my_net = my_net.eval()
    if cuda:
        my_net = my_net.cuda()

    data_iter = iter(data_loader)
    data = data_iter.next()
    img, _ = data

    batch_size = len(img)

    input_img = torch.FloatTensor(batch_size, n_channels, image_size, image_size)

    if cuda:
        img = img.cuda()
        input_img = input_img.cuda()
    # test
    input_img.resize_as_(img).copy_(img)
    inputv_img = Variable(input_img)

    _, _, _, _, rec_img = my_net(inputv_img,Issource,Rec_scheme,1.0)
    vutils.save_image(input_img, 'svhn_real.png', nrow=8)
    vutils.save_image(rec_img.data, 'svhn_rec.png', nrow=8)

    print 'done'

Rec_scheme = 'all'#'shared',private, all
Issource = 'target'#target, source
# mode = 'mnist_trn','mnist_tst','svhn_trn','svhn_tst'
for epoch in xrange(0,145):
    print epoch
    mode = 'mnist_trn'
    rec_image(epoch,mode,Rec_scheme,'target')
    mode = 'mnist_tst'
    rec_image(epoch, mode, Rec_scheme, 'target')
    mode = 'svhn_trn'
    rec_image(epoch, mode, Rec_scheme, 'source')
    mode = 'svhn_tst'
    rec_image(epoch, mode, Rec_scheme, 'source')