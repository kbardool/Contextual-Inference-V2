source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs              400   \
    --steps_in_epoch       64   \
    --last_epoch          296   \
    --batch_size            1   \
    --lr               0.0001   \
    --val_steps             8   \
    --fcn_arch          fcn32   \
    --fcn_layers          all   \
    --fcn_losses         fcn_BCE_loss \
    --mrcnn_logs_dir     train_mrcnn_coco \
    --fcn_logs_dir       train_fcn32_bce\\fcn20181209T0000 \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
##  --new_log_folder     \
    --sysout             file   
    
source deactivate

## 12-09-2018 start epoch 0 ended epoch 100 loss 0.0190139
## 12-10-2018 start epoch 100 ended 296 - loss 0.03522 weight file 0147 