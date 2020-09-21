python checkpointer.py --model 'checkpoint/classification.model' --height 128 --width 128 && python evaluator.py --model 'checkpoint/classification.model' --threshold 0.5

#python checkpointer.py --model 'checkpoint/Reg8.0_white_sides_last.pt' --height 256 --width 256 && python evaluator.py --model 'checkpoint/Reg8.0_white_sides_last.pt' --threshold 0.5
#python checkpointer.py --model 'checkpoint/Reg6.4_white_sides_auroc_best.pt' --height 256 --width 256 && python evaluator.py --model 'checkpoint/Reg6.4_white_sides_auroc_best.pt' --threshold 0.5
