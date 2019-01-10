source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs                3   \
    --steps_in_epoch        2   \
    --last_epoch            0   \
    --batch_size            2   \
    --lr                0.001   \
    --val_steps             2   \
    --fcn_arch           fcn8   \
    --fcn_layers          all   \
    --mrcnn_logs_dir   train_mrcnn_coco \
    --fcn_logs_dir     train_fcn8_coco \
    --mrcnn_model        last   \
    --fcn_model          init   \
    --opt             adagrad   \
    --sysout            screen   \
    --new_log_folder  
source deactivate
