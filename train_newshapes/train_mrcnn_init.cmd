source activate TFG
python ../mrcnn/train_nshapes_mrcnn.py  \
    --epochs             3000  \
    --steps_in_epoch      128  \
    --last_epoch            0  \
    --batch_size           16  \
    --lr                0.001  \
    --mrcnn_logs_dir     train_mrcnn_newshapes  \
    --fcn_logs_dir       train_fcn_newshapes    \
    --mrcnn_model        coco  \
    --val_steps            32  \
    --opt                adam  \
    --sysout             all  \
    --new_log_folder  

source deactivate    
