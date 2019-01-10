source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs             1000   \
    --steps_in_epoch       32   \
    --last_epoch         1674   \
    --batch_size            1   \
    --lr                0.001   \
    --val_steps             8   \
    --fcn_arch           fcn8   \
    --fcn_layers          all   \
    --mrcnn_logs_dir     train_mrcnn_coco \
    --fcn_logs_dir       train_fcn8_coco_adam \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt                adam   \
    --sysout             file   
source deactivate

##  --new_log_folder  
##    --lr               0.0001   \
##  11-11-2018   last epoch 786
##  13-11-2018   start 786  1000 epochs ended Early Stopping at 1100 error 0.0000440
##  15-11-2018   start 1100 1000 epochs ended due to machine stop error ~ 0.0000456
##  16-11-2018   start 1674 1000 start with LR 0.001 
##