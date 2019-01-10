source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs               15  \
    --steps_in_epoch       12   \
    --last_epoch            0   \
    --batch_size            2   \
    --lr                  0.01   \
    --val_steps             4   \
    --mrcnn_logs_dir   train_mrcnn_coco \
    --fcn_logs_dir     train_fcn_coco_adagrad \
    --mrcnn_model              last   \
    --fcn_model          init   \
    --opt             adagrad   \
    --sysout             file   \
    --new_log_folder  
source deactivate
