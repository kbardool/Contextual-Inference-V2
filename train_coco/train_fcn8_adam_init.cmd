source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs               10   \
    --steps_in_epoch       32   \
    --last_epoch            0   \
    --batch_size            1   \
    --lr              0.000001   \
    --val_steps             8   \
    --fcn_arch           fcn8   \
    --fcn_layers          all   \
    --mrcnn_logs_dir   train_mrcnn_coco \
    --fcn_logs_dir     train_fcn8_coco_adam \
    --mrcnn_model        last   \
    --fcn_model          init   \
    --opt                adam   \
    --sysout             file    \
    --new_log_folder  
source deactivate

##    --opt             adagrad   \
