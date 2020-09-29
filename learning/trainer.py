import numpy as np
import torch
from utils import AverageMeter, accuracy, auroc
from learning.smoothing import LabelSmoothing
from learning.mixup import MixUpWrapper, NLLMultiLabelSmooth
from learning.cutmix import cutmix

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, data_loader, epoch, args, result_dict):
        total_loss = 0
        count = 0
        
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        if args.mixup > 0.0:
            self.criterion = NLLMultiLabelSmooth(args.label_smooth)
            data_loader = MixUpWrapper(args.num_classes, args.mixup, data_loader)
        elif args.label_smooth > 0.0:
            self.criterion = LabelSmoothing(args.label_smooth)

        output_list = []
        labels_list =[]
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            if args.cutmix_alpha > 0:
                r = np.random.rand(1)
                if r < args.cutmix_prob:
                    outputs, loss = cutmix(args, self.model, self.criterion, inputs, labels)
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                        # has dot product
                    outputs, dp = outputs
                else:
                    dp = None

                loss = self.criterion(outputs, labels)
                #FIXME
            
            if len(labels.size()) > 1:
                labels = torch.argmax(labels, axis=1)

            prec1, prec3 = accuracy(outputs.data, labels, args.num_classes,topk=(1, 3))
            
            output_list.append(outputs)
            labels_list.append(labels)

            if dp is not None:
                dp_loss = 0.1 * torch.abs(dp.mean())
                loss = loss + dp_loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.tolist()
            count += labels.size(0)

            if batch_idx % args.log_interval == 0:
                if args.mixup > 0.0:
                    _s = str(len(str(len(data_loader.dataloader.sampler))))
                    ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.dataloader.sampler),\
                     100 * count / len(data_loader.dataloader.sampler)),
                    'train_loss: {: >4.2e}'.format(total_loss / count),
                    'train_accuracy : {:.2f}%'.format(top1.avg)
                    ]
                else:
                    _s = str(len(str(len(data_loader.sampler))))
                    ret = [
                        ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.sampler), 100 * count / len(data_loader.sampler)),
                        'train_loss: {: >4.2e}'.format(total_loss / count),
                        'train_accuracy : {:.2f}%'.format(top1.avg)
                    ]
                print(', '.join(ret))
        concated_outputs = torch.cat(output_list, dim = 0)
        concated_labels = torch.cat(labels_list, dim = 0)
        AUROC = auroc(concated_outputs.data, concated_labels)
        
        self.scheduler.step()
        result_dict['train_loss'].append(losses.avg)
        result_dict['train_acc'].append(top1.avg)
        result_dict['train_auroc'].append(AUROC.item())

        return result_dict