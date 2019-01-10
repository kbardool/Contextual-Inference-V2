source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs             2000   \
    --steps_in_epoch       64   \
    --last_epoch          218   \
    --batch_size            1   \
    --lr               0.0001   \
    --val_steps             8   \
    --fcn_arch          fcn8l2  \
    --fcn_layers         all    \
    --fcn_losses         fcn_BCE_loss \
    --mrcnn_logs_dir     train_mrcnn_coco_subset \
    --fcn_logs_dir       train_fcn8_l2_bce_subset \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --sysout             file   \
    --new_log_folder
    
source deactivate

#
# --coco_classes:
#  appliance: 78 - 82   kitchen: 44 - 51   sports: 34 - 43       indoor: 10 -15
##---------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - all layers - w/ fcn_BCE_loss: Folder  weight decay: 1.0e-6 (I think loadAnns = 'allclasses)
## folder train_fcn8_l2_bce\fcn20181216T0000
##---------------------------------------------------------------------------------------------------------------
## 16-12-2018  start   0, 500 epochs LR  0.0001 end machine shutdown @epoch 218  err 0.01271     wght file 0154
## 17-12-2018  start 218, 500 epochs LR 0.00005 end early stop       @epoch 449  err 0.005845    wght file 300
## xx-xx-2018  start    , xxx epochs LR         end                  @epoch      err             wght file 01
##
##--------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - All layers - w/ fcn_BCE_loss: Training on coco subset with loadAnns = 'active_only'
##--------------------------------------------------------------------------------------------------------------
## 21-12-2018  start   0, 2000 epochs LR  0.0001 end disk space       @epoch   25 err 0.0131407  wght file 0010
## 21-12-2018  start  25, 2000 epochs LR  0.0001 end machine shutdown @epoch  583 err 0.0037354  wght file 0547  
## 22-12-2018  start 583, 2000 epochs LR  0.0001 end                  @epoch      err            wght file 01
