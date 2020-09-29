import torch
import os, sys
import torch
import torch.nn as nn
import torchvision

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH + '/../..')

from option import get_args
from utils import get_model

torch.backends.cudnn.benchmark = True
args = get_args()
torch.manual_seed(args.seed)
shape = (128,128,3)  

model = get_model(args, shape, args.num_classes)



model.load(r"C:\Users\esuh\Desktop\Project\COI-Carrier\carrier-of-tricks-for-classification-pytorch\checkpoint\{}\best_model".format(args.checkpoint_name, args.checkpoint_name))
model.eval()
example = torch.rand(1, 3, 128, 128) # network input size example ( 1 batch )

# print(model(torch.ones(1,3,128,128)))

traced_script_module = torch.jit.trace(model, example)
jit_path= r"C:\Users\esuh\Desktop\Project\COI-Carrier\carrier-of-tricks-for-classification-pytorch\jit_trace\{}.pt".format(args.checkpoint_name)
traced_script_module.save(jit_path)
print("jit saved succesfully")

model2 = torch.jit.load(jit_path)
# print(model2(torch.ones(1,3,128,128)))
