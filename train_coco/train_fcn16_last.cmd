source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs               20   \
    --steps_in_epoch       64   \
    --last_epoch            0   \
    --batch_size            1   \
    --lr               0.0001   \
    --val_steps            16   \
    --fcn_arch          fcn16   \
    --fcn_layers          all   \
    --mrcnn_logs_dir   train_mrcnn_coco \
    --fcn_logs_dir     train_fcn16_coco \
    --mrcnn_model        last   \
    --fcn_model          init   \
    --opt             adagrad   \
    --sysout             file    \
    --new_log_folder  
source deactivate
