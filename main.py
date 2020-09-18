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

def main():
    args = get_args()
    torch.manual_seed(args.seed)

    shape = (256,256,3)    

    """ define dataloader """
    train_loader, valid_loader, test_loader = make_dataloader(args)

    """ define model architecture """
    model = get_model(args, shape, args.num_classes)

    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        model = model.cuda() 
    else:
        raise ValueError('CPU training is not supported')

    """ define loss criterion """
    criterion = nn.CrossEntropyLoss().cuda()

    """ define optimizer """
    optimizer = make_optimizer(args, model)

    """ define learning rate scheduler """
    scheduler = make_scheduler(args, optimizer)
    
    """ define trainer, evaluator, result_dictionary """
    result_dict = {'args':vars(args), 'epoch':[], 'train_loss' : [], 'train_acc' : [], 'train_auroc' : [], 'val_loss' : [], 'val_acc' : [], 'val_auroc' : [], 'test_acc':[], 'test_auroc' : []}
    trainer = Trainer(model, criterion, optimizer, scheduler)
    evaluator = Evaluator(model, criterion)

    if args.evaluate:
        """ load model checkpoint """
        model.load("best_model")
        result_dict = evaluator.test(test_loader, args, result_dict, True)
        # print(model(torch.ones(1,3,256,256).cuda()))

        model.load("last_model")
        result_dict = evaluator.test(test_loader, args, result_dict, False)


    else:
        evaluator.save(result_dict)

        best_val_acc = 0.0
        best_val_auroc = 0.0
        """ define training loop """
        tolerance = 0

        for epoch in range(args.epochs):
            result_dict['epoch'] = epoch
            result_dict = trainer.train(train_loader, epoch, args, result_dict)
            result_dict = evaluator.evaluate(valid_loader, epoch, args, result_dict)

            tolerance +=1
            print("tolerance: ",tolerance)

            # if result_dict['val_acc'][-1] > best_val_acc:
            if result_dict['val_auroc'][-1] > best_val_auroc:
                tolerance = 0
                print("{} epoch, best epoch was updated! {}%".format(epoch, result_dict['val_auroc'][-1]))
                best_val_auroc = result_dict['val_auroc'][-1]
                model.save(checkpoint_name='best_model')

            evaluator.save(result_dict)
            plot_learning_curves(result_dict, epoch, args)

            if tolerance > 45:
                break

        result_dict = evaluator.test(test_loader, args, result_dict, False)
        evaluator.save(result_dict)

        """ save model checkpoint """
        model.save(checkpoint_name='last_model')

        """ calculate test accuracy using best model """
        model.load(checkpoint_name='best_model')
        result_dict = evaluator.test(test_loader, args, result_dict,True)
        evaluator.save(result_dict)

    print(result_dict)
if __name__ == '__main__':
    main()