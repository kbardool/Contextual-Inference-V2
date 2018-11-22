source activate TFG
python train_coco_mrcnn.py      \
	--epochs               40   \
	--steps_in_epoch       10   \
	--last_epoch           75   \
    --batch_size            4   \
    --lr                0.001   \
    --val_steps             4   \
    --logs_dir     train_mrcnn_coco \
	--mrcnn_model        last   \
    --opt             adagrad   \
    --sysout             file   \
    --new_log_folder  
        
source deactivate
