source activate TFG
python ../mrcnn/train_nshapes_mrcnn.py  \
    --epochs               200  \
    --last_epoch           0    \
    --steps_in_epoch       64   \
    --val_steps            16   \
    --batch_size           8    \
    --lr                   0.0001  \
    --mrcnn_logs_dir       train_mrcnn  \
    --mrcnn_model          last  \
    --mrcnn_layers         mrcnn rpn fpn  \
    --opt                  adam   \
    --scale_factor         1   \
    --toy_dataset          newshapes2 \
    --sysout               all  \
    --new_log_folder  

source deactivate    
