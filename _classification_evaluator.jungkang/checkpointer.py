import time
import torch
import argparse
import numpy as np
import cv2
from skimage.io import imread
from skimage import transform
import torchvision
from torchvision import transforms as T
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import get_model
from PIL import Image

class WildDataset(torch.utils.data.Dataset):

    def __init__(self, args, transform = None):
        self.args = args
        with open(args.annotation, 'r') as anno:
            self.anno_list = anno.read().splitlines()[1:]
        self.transform = transform
        
    def __len__(self):
        return len(self.anno_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = self.anno_list[idx].split(';')[0]
        img_name = '{}/{}'.format(self.args.image, fname)
        image = imread(img_name)
        image = Image.fromarray(image)
        
        if self.transform:
            image  = self.transform(image)
        return image, fname


class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
    def __call__(self, image):
        new_h , new_w = self.output_size
        new_h, new_w =int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img *255.0

class ToTensor(object):
    def __call__(self,image):
        image = image.transpose((2,0,1))
        tensor = torch.from_numpy(image)
        return tensor.contiguous().float()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit_model', type=str, default='checkpoint/classification.model')
    parser.add_argument('--model', type=str, default='')   
    parser.add_argument('--annotation', type=str, default='dataset/a415f-white/side_annotation.csv')
    parser.add_argument('--image', type=str, default='dataset/a415f-white/image')
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--jit', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default=r'C:\Users\esuh\Desktop\Project\COI-Carrier\carrier-of-tricks-for-classification-pytorch\checkpoint')
    parser.add_argument('--checkpoint_name', type=str, default='') 
    parser.add_argument('--model_size', type =str, default = '3.2GF')

    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--norm', default='batchnorm', choices=('batchnorm', 'evonorm'),
                        help='normalization to use (batchnorm | evonorm)')
    parser.add_argument('--zero_gamma', action='store_true', default=False)
    parser.add_argument('--batch_size', type = int, default = 16)
    
    

    

    args = parser.parse_args()
    return args

def to_tensor(image, size=(128,128)):
    h, w = size
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 0 ,1))
    image = np.expand_dims(image, axis=0)
    tensor = torch.from_numpy(image)
    return tensor.contiguous().float()  / 255.

def checkpoint(args):
    torch.manual_seed(args.seed)
    shape = (args.height,args.width,3)
    torch.backends.cudnn.benchmark = True
    
    if args.jit:
        model = torch.jit.load(args.jit_model).cuda()
    else:
        model = get_model(args, shape, args.num_classes)
        if torch.cuda.device_count() >= 1:
            print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
            model = model.cuda() 
        else:
            raise ValueError('CPU training is not supported')
        
        model.load("best_model")

        
    model.eval()
    # print(model(torch.ones(1,3,256,256).cuda()))
    softmax = torch.nn.Softmax(dim=1)
    if args.jit:
        csv = open('{}.csv'.format(args.jit_model), 'w')
    else:
        csv = open('checkpoint/{}.csv'.format(args.checkpoint_name), 'w')
    csv.write('patch_filename;ok_prob;ng_prob;evaluation_time;total_time\n')

    transform = T.Compose([
        T.Resize((256,256)),
        T.ToTensor()

        ]) 
    
    # T.ToTensor()
    # Rescale((256,256)),
    #     ToTensor()
        #     T.Resize((256,256)),
        # T.ToTensor()
    
    wild_dataset = WildDataset(args = args, transform = transform)
    dataloader = torch.utils.data.DataLoader(wild_dataset,batch_size = args.batch_size, shuffle = False, num_workers = 0)



    # dataloader
    # with torch.no_grad():
    total_time_start = time.time()
    eval_time_sum = 0
    with torch.no_grad():
        for i_batch, (x, fname) in enumerate(dataloader):
            x = x.cuda()
            x= x*255.0
            eval_time_start = time.time()
            y_ = softmax(model(x))  # probability: tensor([[9.9954e-01, 4.5962e-04]], device='cuda:0', grad_fn=<SoftmaxBackward>)
            eval_time = time.time() - eval_time_start
            eval_time_sum += eval_time
            print('{}/{}'.format(i_batch*args.batch_size, len(dataloader.dataset)))
            for j in range(len(fname)):
                ok, ng = y_.cpu().detach().numpy()[j]
                total_time = time.time() - total_time_start
                csv.write('{};{};{};{};{}\n'.format(fname[j],ok,ng,eval_time_sum,total_time))
        print(eval_time_sum)
        print(total_time)


def checkpoint_origin(args):
    torch.manual_seed(args.seed)
    shape = (args.height,args.width,3)
    torch.backends.cudnn.benchmark = True
    
    if args.jit:
        model = torch.jit.load(args.jit_model).cuda()
    else:
        model = get_model(args, shape, args.num_classes)
        if torch.cuda.device_count() >= 1:
            print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
            model = model.cuda() 
        else:
            raise ValueError('CPU training is not supported')
        
        model.load("best_model")
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    if args.jit:
        csv = open('{}.csv'.format(args.model), 'w')
    else:
        csv = open('checkpoint/{}.csv'.format(args.checkpoint_name), 'w')
 
    csv.write('patch_filename;ok_prob;ng_prob;evaluation_time;total_time\n')

    with open(args.annotation, 'r') as anno:
        for line in anno.read().splitlines()[1:]:
            fname = line.split(';')[0]
            f = '{}/{}'.format(args.image, fname)

            total_time_start = time.time()
            x = to_tensor(imread(f), (args.height, args.width)).cuda()
            eval_time_start = time.time()
            y_ = softmax(model(x))  # probability: tensor([[9.9954e-01, 4.5962e-04]], device='cuda:0', grad_fn=<SoftmaxBackward>)
            eval_time = time.time() - eval_time_start
            ok, ng = y_.cpu().detach().numpy()[0]
            total_time = time.time() - total_time_start
            csv.write('{};{};{};{};{}\n'.format(fname,ok,ng,eval_time,total_time))
    


if __name__ == '__main__':
    args = get_args()
    with torch.cuda.device(0):
        checkpoint(args)
