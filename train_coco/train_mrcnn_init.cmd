source activate TFG
python ../mrcnn/train_coco_mrcnn.py  \
    --epochs              100 \
    --steps_in_epoch       64 \
    --val_steps            16 \
    --last_epoch            0 \
    --batch_size            1 \
    --lr               0.0001 \
    --mrcnn_logs_dir     train_mrcnn_coco_subset  \
    --mrcnn_model        coco   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --sysout             file   \
    --new_log_folder  
        
source deactivate