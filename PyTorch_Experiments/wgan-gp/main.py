from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from optimizers import *
from adabound import AdaBound
from torch.optim import SGD, Adam
from fid_score import calculate_fid_given_paths
import pandas as pd
BATCH_SIZE=64
LAMBDA = 10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='cifar10', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, default='./',help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True)
parser.add_argument('--partial', default=1.0/4.0, type=float)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
parser.add_argument('--beta2', default=0.999, type=float, help='Beta2')
parser.add_argument('--eps',default=1e-8, type=float, help='eps')
parser.add_argument('--eps_sqrt', default=0.0, type=float, help='eps_sqrt')
parser.add_argument('--final_lr', default=1e-2, type=float, help='final_lr')
parser.add_argument('--Train', action = 'store_true')
parser.add_argument('--run', default=0, type=int, help='runs')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

opt = parser.parse_args()
opt.outf = opt.optimizer + '-wgan' + '-betas{}-{}'.format(opt.beta1, opt.beta2) + '-eps{}-{}'.format(opt.eps, opt.eps_sqrt) \
           + '-final-lr{}'.format(opt.final_lr) + '-run{}'.format(str(opt.run)) + '-partial-{}'.format(opt.partial) + '-clip-{}'.format(opt.clip_value)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

#device = torch.device("cuda:{}".format(os.environ['CUDA_VISIBLE_DEVICES']) if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.view(-1,1,1,1)#alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 64, 64)
    alpha = alpha.cuda()#(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #if use_cuda:
    interpolates = interpolates.cuda()#gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).cuda()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.linear = nn.Linear(4*4*8*ndf,1)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).cuda()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()#, device=device)
real_label = 1
fake_label = 0

# setup optimizer
# setup optimizer
opt.optimizer = opt.optimizer.lower()
if opt.optimizer == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
elif opt.optimizer == 'adamw':
    optimizerD = AdamW(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optimizerG = AdamW(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
elif opt.optimizer == 'fromage':
    optimizerD = Fromage(netD.parameters(), lr=opt.lr)
    optimizerG = Fromage(netG.parameters(), lr=opt.lr)
elif opt.optimizer == 'adabelief':
    optimizerD = AdaBelief(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optimizerG = AdaBelief(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
elif opt.optimizer == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr)
    optimizerG = torch.optim.SGD(netG.parameters(), lr=opt.lr)
elif opt.optimizer == 'rmsprop':
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lr)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr)
elif opt.optimizer == 'adabound':
    optimizerD = AdaBound(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps,
                          final_lr=opt.final_lr)
    optimizerG = AdaBound(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps,
                          final_lr = opt.final_lr)
elif opt.optimizer == 'yogi':
    optimizerD = Yogi(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
    optimizerG = Yogi(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
elif opt.optimizer == 'radam':
    optimizerD = RAdam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
    optimizerG = RAdam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),eps=opt.eps)
elif opt.optimizer == 'msvag':
    optimizerD = MSVAG(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerG = MSVAG(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# convert all training data into png format
real_folder = 'all_real_imgs'
if not os.path.exists(real_folder):
    os.mkdir(real_folder)
    for i in range(len(dataset)):
        vutils.save_image(dataset[i][0], real_folder + '/{}.png'.format(i), normalize=True)

fake_folder = 'all_fake_imgs' + opt.outf
if not os.path.exists(fake_folder):
    os.mkdir(fake_folder)

FIDs = []
fake_images_number = 1000

print(opt.Train)

if opt.Train == True:

    for epoch in range(opt.niter):
        print('Epoch {}'.format(epoch))
        
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].cuda()
            batch_size = real_cpu.size(0)

            # Configure input
            real_imgs = real_cpu#Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()

            # Sample noise as netG input
            z = torch.randn(batch_size, nz, 1, 1).cuda()#, device=device)#Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = netG(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))

            loss_D.backward()

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
            gradient_penalty.backward()

            optimizerD.step()

            # Clip weights of netD
            #for p in netD.parameters():
            #    p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the netG every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizerG.zero_grad()

                # Generate a batch of images
                gen_imgs = netG(z)
                # Adversarial loss
                loss_G = -torch.mean(netD(gen_imgs))

                loss_G.backward()
                optimizerG.step()

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if True:
    batch_size = opt.batchSize
    netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (opt.outf, opt.niter-1)))

    # test netG, and calculate FID score
    netG.eval()
    for i in range(fake_images_number):
        noise = torch.randn(batch_size, nz, 1, 1).cuda()
        fake = netG(noise)
        for j in range(fake.shape[0]):
            vutils.save_image(fake.detach()[j,...], fake_folder + '/{}.png'.format(j + i * batch_size), normalize=True)
    netG.train()

    # calculate FID score
    fid_value = calculate_fid_given_paths([real_folder, fake_folder],
                                          opt.batchSize//2,
                                         True)
    FIDs.append(fid_value)

    print('FID: {}'.format(fid_value))

    df = pd.DataFrame(FIDs)
    df.to_csv('FID_{}.csv'.format(opt.outf))

