source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs              100   \
    --steps_in_epoch       64   \
    --last_epoch            0   \
    --batch_size            1   \
    --lr                0.001   \
    --val_steps             8   \
    --fcn_arch          fcn32   \
    --fcn_layers          all   \
    --mrcnn_logs_dir     train_mrcnn_coco \
    --fcn_logs_dir       train_fcn32_bce  \
    --mrcnn_model        last   \
    --fcn_model          init   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --new_log_folder     \
    --sysout             file
    
source deactivate
