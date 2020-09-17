import os
import json
import torch
from utils import AverageMeter, accuracy, auroc

class Evaluator():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        self.save_path = os.path.join(self.model.checkpoint_dir, self.model.checkpoint_name, 'result_dict.json')
        if not os.path.exists(os.path.join(self.model.checkpoint_dir, self.model.checkpoint_name)):
            os.makedirs(os.path.join(self.model.checkpoint_dir, self.model.checkpoint_name))

    def worst_result(self):
        ret = { 
            'loss': float('inf'),
            'accuracy': 0.0
         }
        return ret
        
    def result_to_str(self, result):
        ret = [
            'epoch: {epoch:0>3}',
            'loss: {loss: >4.2e}'
        ]
        for metric in self.evaluation_metrics:
            ret.append('{}: {}'.format(metric.name, metric.fmtstr))
        return ', '.join(ret).format(**result)
    
    def save(self, result):
        with open(self.save_path, 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def load(self):
        result = self.worst_result
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                try:
                    result = json.loads(f.read())
                except:
                    pass
        return result

    def evaluate(self, data_loader, epoch, args, result_dict):
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            output_list = []
            labels_list =[]
            for batch_idx, (inputs, labels) in enumerate(data_loader):

                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                prec1, prec3 = accuracy(outputs.data, labels, args.num_classes,topk=(1, 3))
             
                output_list.append(outputs)
                labels_list.append(labels)
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))

                        
        print('----Validation Results Summary----')
        concated_outputs = torch.cat(output_list, dim = 0)
        concated_labels = torch.cat(labels_list, dim = 0)

        AUROC = auroc(concated_outputs.data, concated_labels)
        print('Epoch: [{}] Top-1 accuracy: {:.2f}%, AUROC: {:.3f}%'.format(epoch, top1.avg, AUROC.item()))

        result_dict['val_loss'].append(losses.avg)
        result_dict['val_acc'].append(top1.avg)
        result_dict['val_auroc'].append(AUROC.item())

        return result_dict

    def test(self, data_loader, args, result_dict, is_best):
        top1 = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            output_list = []
            labels_list =[]
            for batch_idx, (inputs, labels) in enumerate(data_loader):

                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                
                prec1, prec3 = accuracy(outputs.data, labels, args.num_classes,topk=(1, 3))

                output_list.append(outputs)
                labels_list.append(labels)

                top1.update(prec1.item(), inputs.size(0))


        if is_best:
            print('----Test Set Results with the best Summary----')
        else:
            print('----Test Set Results with the last Summary----')

        concated_outputs = torch.cat(output_list, dim = 0)
        concated_labels = torch.cat(labels_list, dim = 0)

        AUROC = auroc(concated_outputs.data, concated_labels)

        print('Top-1 accuracy: {:.2f}%'.format(top1.avg))
        print('AUROC: {:.3f}%'.format(AUROC.item()))


        result_dict['test_acc'].append(top1.avg)
        result_dict['test_auroc'].append(AUROC.item())

        return result_dict
    
