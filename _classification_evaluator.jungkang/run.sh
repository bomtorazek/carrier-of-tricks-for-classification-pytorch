#python checkpointer.py \
#    --jit_model 'checkpoint/Reg6.4_white_sides_auroc.pt' \
#    --batch_size 16
#python evaluator.py \
#    --pred 'checkpoint/Reg6.4_white_sides_auroc.pt.csv' \
#    --true 'dataset/a415f-white/side_annotation.csv' \
#    --threshold 0.5


python checkpointer.py \
    --jit_model 'checkpoint/classification.model' \
    --batch_size 16
python evaluator.py \
    --pred 'checkpoint/classification.model.csv' \
    --true 'dataset/a415f-white/side_annotation.csv' \
    --threshold 0.5
