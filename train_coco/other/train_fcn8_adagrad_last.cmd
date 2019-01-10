source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs              100   \
    --steps_in_epoch       32  \
    --last_epoch         2697   \
    --batch_size            1   \
    --lr               0.0001   \
    --val_steps             8    \
    --fcn_arch           fcn8   \
    --fcn_layers          all   \
    --mrcnn_logs_dir   train_mrcnn_coco \
    --fcn_logs_dir     train_fcn8_coco_adagrad  \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt             adagrad   \
    --sysout            file    \
##  --new_log_folder  
source deactivate
##    --lr               0.0001   \
