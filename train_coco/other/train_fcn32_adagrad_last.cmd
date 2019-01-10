source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs               17   \
    --steps_in_epoch       32   \
    --last_epoch           53   \
    --batch_size            2   \
    --lr                 0.01   \
    --val_steps             8   \
    --fcn_arch          fcn32   \
    --mrcnn_logs_dir   train_mrcnn_coco \
    --fcn_logs_dir     train_fcn_coco_adagrad \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt             adagrad   \
    --sysout             file
source deactivate
##  --new_log_folder  \

## 
## source activate TFG
## python train_coco_fcn.py  \
##   --epochs          100   \
##   --steps_in_epoch  400   \
##   --last_epoch        0   \
##   --batch_size       32   \
##   --val_steps        16   \
##   --lr             0.01   \
##   --opt          rmsprop  \
##   --logs_dir     coco_fcn \
##   --model        coco     \
##   --fcn_model    last
## source deactivate
## 
