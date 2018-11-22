source activate TFG
python train_coco_mrcnn.py  \
    --epochs               10 \
    --steps_in_epoch       10 \
    --last_epoch            0 \
    --batch_size            2 \
    --lr                0.001 \
    --mrcnn_logs_dir train_mrcnn_coco  \
    --mrcnn_model        coco         
    --val_steps             4   \
    --opt             adagrad   \
    --sysout             file   \
    --new_log_folder  
        
source deactivate