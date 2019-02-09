source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs             2000   \
    --steps_in_epoch       64   \
    --last_epoch            0   \
    --batch_size            1   \
    --lr               0.0001   \
    --val_steps             8   \
    --fcn_arch          fcn8l2  \
    --fcn_layers         all    \
    --fcn_losses         fcn_BCE_loss \
    --mrcnn_logs_dir     train_mrcnn_coco_subset \
    --fcn_logs_dir       train_fcn8_l2_bce_subset \
    --mrcnn_model        last   \
    --fcn_model          init   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --new_log_folder     \
    --sysout             file
    
source deactivate

#
# --coco_classes:
#  appliance: 78 - 82   kitchen: 44 - 51   sports: 34 - 43       indoor: 10 -15
##----------------------------------------------------------------------------------------------------
## running over fcn_BCE_loss: Folder  weight decay: 1.0e-6
##----------------------------------------------------------------------------------------------------
## 15-12-2018  start   0, 500 epochs LR  0.0001 end machine shutdown @epoch 218  err 0.01271  wght file 0154
## xx-12-2018  start    , 500 epochs LR         end                  @epoch      err          wght file
## xx-12-2018  start   0, 500 epochs LR         end                  @epoch      err          wght file 01
##
##----------------------------------------------------------------------------------------------------
## running over fcn_BCE_loss: Training on coco subset with loadAnns = 'active_only'
##----------------------------------------------------------------------------------------------------
## 21-12-2018  start   0, 500 epochs LR  0.0001 end                  @epoch      err          wght file      
## xx-12-2018  start    , 500 epochs LR         end                  @epoch      err          wght file
## xx-12-2018  start   0, 500 epochs LR         end                  @epoch      err          wght file 01
