import torch
import os, sys
import torch
import torch.nn as nn
import torchvision

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH + '/../..')

from option import get_args
from learning.trainer import Trainer
from learning.evaluator import Evaluator
from utils import get_model, make_optimizer, make_scheduler, make_dataloader, plot_learning_curves


torch.backends.cudnn.benchmark = True
args = get_args()
torch.manual_seed(args.seed)
shape = (256,256,3)  

model = get_model(args, shape, args.num_classes)



model.load(r"C:\Users\esuh\Desktop\Project\COI-Carrier\carrier-of-tricks-for-classification-pytorch\checkpoint\{}\best_model".format(args.checkpoint_name, args.checkpoint_name))
model.eval()
example = torch.rand(1, 3, 256, 256) # network input size example ( 1 batch )

# print(model(torch.ones(1,3,256,256)))
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(r"C:\Users\esuh\Desktop\Project\COI-Carrier\carrier-of-tricks-for-classification-pytorch\jit_trace\{}.pt".format(args.checkpoint_name))

print("jit saved succesfully")
