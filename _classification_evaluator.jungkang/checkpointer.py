import time
import torch
import argparse
import glob
from skimage.io import imread
from skimage import transform
from PIL import Image
from torchvision import transforms as T


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit_model', type=str, default='checkpoint/Reg6.4_white_sides_auroc.pt')
    parser.add_argument('--image', type=str, default='dataset/a415f-white/image/')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type = int, default = 16)
    args = parser.parse_args()
    return args

def checkpoint(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True # FIXME

    transform = T.Compose([
        T.ToTensor()
    ]) 
    wild_dataset = WildDataset(args=args, transform=transform)
    dataloader = torch.utils.data.DataLoader(wild_dataset,batch_size=args.batch_size, shuffle = False, num_workers = 0)

    model = torch.jit.load(args.jit_model).cuda()
    softmax = torch.nn.Softmax(dim=1)
    model.eval()

    csv = open('{}{}.csv'.format('checkpoint/',args.jit_model.split('/')[-1]), 'w')
    csv.write('patch_filename;ok_prob;ng_prob\n')
    total_time_start = time.time()
    eval_time = 0
    with torch.no_grad():
        for i_batch, (x, fname) in enumerate(dataloader):
            x = x.cuda()
            eval_time_start = time.time()
            y_ = softmax(model(x))
            eval_time += time.time() - eval_time_start
            if i_batch % 50 ==0:
                print('{}/{}'.format(i_batch*args.batch_size, len(dataloader.dataset)))
            for j in range(len(x)):
                ok, ng = y_.cpu().detach().numpy()[j]
                csv.write('{};{};{}\n'.format(fname[j], ok, ng))
    total_time = time.time() - total_time_start
    print('evaluation time: {:.2f}'.format(eval_time))
    print('total time: {:.2f}'.format(total_time))
    print('batch size: {}'.format(args.batch_size))

class WildDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform = None):
        self.args = args
        self.anno_list = [ f.split('\\')[-1] for f in sorted(glob.glob('{}/*.png'.format(args.image))) ]
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
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        new_h, new_w =int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img * 255.0

class ToTensor(object):
    def __call__(self,image):
        image = image.transpose((2,0,1))
        tensor = torch.from_numpy(image)
        return tensor.contiguous().float()


if __name__ == '__main__':
    args = get_args()
    with torch.cuda.device(0):
        checkpoint(args)
