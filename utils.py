import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
osp = os.path

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms as T
from vision.dataset import SingleImageClassificationDataset
from vision.transform import *

from network.resnet import *
from network.efficientnet import *
from network.regnet import *
from network.anynet import *
from learning.lr_scheduler import GradualWarmupScheduler
from learning.radam import RAdam
from learning.randaug import RandAugment

def get_model(args, shape, num_classes):
    if 'ResNet' in args.model:
        model = eval(args.model)(
            shape,
            num_classes,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            pretrained=args.pretrained,
            pretrained_path=args.pretrained_path,
            norm=args.norm,
            zero_init_residual=args.zero_gamma
        )#.cuda(args.gpu)
    elif 'RegNet' in args.model:
        model = eval(args.model)(
            shape,
            1000,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name
        )#.cuda(args.gpu)
        pt_ckpt = torch.load('pretrained_weights/RegNetY-3.2GF_dds_8gpu.pyth', map_location="cpu")
        model.load_state_dict(pt_ckpt["model_state"])
        model.head = AnyHead(w_in=model.prev_w, nc=num_classes)#.cuda(args.gpu)
    elif 'EfficientNet' in args.model:
        model = eval(args.model)(
            shape,
            1000,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name
        )#.cuda(args.gpu)
        pt_ckpt = torch.load('pretrained_weights/EN-B4_dds_8gpu.pyth', map_location="cpu")
        model.load_state_dict(pt_ckpt["model_state"])
        model.head = EffHead(w_in=model.prev_w, w_out=model.head_w, nc=num_classes)#.cuda(args.gpu)
    else:
        raise NameError('Not Supportes Model')
    
    return model


def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RADAM':
        optimizer_function = RAdam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    else:
        raise NameError('Not Supportes Optimizer')

    kwargs['lr'] = args.learning_rate
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif args.decay_type == 'step_warmup':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=scheduler
        )
    elif args.decay_type == 'cosine_warmup':
        cosine_scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=args.epochs//10,
            after_scheduler=cosine_scheduler
        )
    else:
        raise Exception('unknown lr scheduler: {}'.format(args.decay_type))
    
    return scheduler

def make_dataloader(args):
    
    train_trans = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    ])
    
    if args.randaugment:
        #train_trans.transforms.insert(0, RandAugment(3, 5))
        train_trans.transforms.insert(0, RandAugment(args.rand_n, args.rand_m))

    valid_trans = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        ])
    
    test_trans = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        ])

    transdict = {'train': [To3channel(),Resize((256,256)), HFlip(), ToTensor()], 'val': [To3channel(),Resize((256,256)), ToTensor()], 'test':[To3channel(),Resize((256,256)), ToTensor()]}
    

    # for sualab
    if args.sua_data:
        DEFAULT_DATADIR = r'C:\Users\esuh\data\999_project\015_COI_rnd_tf\a415f-white\front\patch_cropped_binary-labeled_for_cls'
        imsets = {'train': osp.join(DEFAULT_DATADIR, r'imageset\single_2class\fold.5-5-4\ratio\100%\trainval.{}-1.txt'.format(args.sua_fold)), 
                'val':osp.join(DEFAULT_DATADIR, r'imageset\single_2class\fold.5-5-4\ratio\100%\test-dev.{}.txt'.format(args.sua_fold)),
                'test':osp.join(DEFAULT_DATADIR, r'imageset\single_2class\fold.5-5-4\ratio\100%\test.1.txt')}
        imdir = osp.join(DEFAULT_DATADIR, 'image')
        antnpath = osp.join(DEFAULT_DATADIR, 'annotation', 'single_2class.json')
        
        datasets = {x: SingleImageClassificationDataset(imdir, antnpath, imsets[x],
                                                transforms=transdict[x]) for x in ['train','val','test']}
        shuffle_dict = {'train' : True, 'val' : True, 'test': False} # 
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle = shuffle_dict[x], num_workers=args.num_workers)
              for x in ['train', 'val', 'test']}

    else:
        trainset = torchvision.datasets.ImageFolder(root=r"C:\Users\esuh\data\999_project\999_intel_dataset\seg_train\seg_train", transform=train_trans) 
        validset = torchvision.datasets.ImageFolder(root=r"C:\Users\esuh\data\999_project\999_intel_dataset\seg_train\seg_train", transform=valid_trans)
        testset = torchvision.datasets.ImageFolder(root=r"C:\Users\esuh\data\999_project\999_intel_dataset\seg_test\seg_test", transform=test_trans)

        np.random.seed(args.seed)
        targets = trainset.targets
        train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets) #No need here
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        dataloaders = {'train': train_loader, 'val':valid_loader, 'test': test_loader}
    

    return dataloaders['train'], dataloaders['val'], dataloaders['test']


def plot_learning_curves(metrics, cur_epoch, args):
    x = np.arange(cur_epoch+1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    # ax1.set_ylabel('loss')
    ax1.set_ylabel('sua_metric')
    ln1 = ax1.plot(x, metrics['train_SUAmetric'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_SUAmetric'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln2+ln3+ln4
    plt.legend(lns, ['Train sua', 'Validation sua', 'Train accuracy','Validation accuracy'])
    plt.tight_layout()
    plt.savefig('{}/{}/learning_curve.png'.format(args.checkpoint_dir, args.checkpoint_name), bbox_inches='tight')
    plt.close('all')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def accuracy(output, target, num_cl,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    if num_cl <3:
        topk = (1,1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # pred >>[[1],[0], ....]
    pred = pred.t() #[[1,0,...]]
    
    # target >> [1,0,...]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def sua_metric(output, target):
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output)
    output = output[:,1]

    threshold = 0
    while 1:
        predict = (output>=threshold).int()
        conf_mat = confusion_matrix(target.cpu(),predict.cpu())
        if (conf_mat.shape) == (2,2):

            tn, fp, fn, tp = conf_mat[0,0], conf_mat[0,1], conf_mat[1,0], conf_mat[1,1]
            print('wow')
        elif conf_mat.shape ==(1,1):
            if predict[0] == 0:
                tn = conf_mat[0,0]
                tp, fp, fn = 0,0,0
            elif predict[1] == 1:
                tp = conf_mat[0,0]
                tn, fp, fn = 0,0,0
            print(tp,tn,fp,fn)

        overkill = fp/(tn+fp+fn+tp)
        underkill = fn / (tn+fp+fn+tp)

        if overkill <= 0.25:
            print(conf_mat)
            return underkill * 100.0

        threshold += 0.01 # 0.0001

        
