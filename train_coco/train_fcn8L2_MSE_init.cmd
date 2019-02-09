source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs             2000   \
    --steps_in_epoch       64   \
    --last_epoch            0   \
    --batch_size            1   \
    --lr              0.00001   \
    --val_steps            16   \
    --fcn_arch           fcn8l2 \
    --fcn_layers         all    \
    --fcn_losses         fcn_MSE_loss \
    --mrcnn_logs_dir     train_mrcnn_coco_subset \
    --fcn_logs_dir       train_fcn8_l2_mse_subset \
    --mrcnn_model        last   \
    --fcn_model          init   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --sysout             file   \
    --new_log_folder    
    
source deactivate

#
# --coco_classes:
#  appliance: 78 - 82   kitchen: 44 - 51   sports: 34 - 43       indoor: 10 -15
#
##--------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - All layers - w/ fcn_MSE_loss: Training on coco subset with loadAnns = 'active_only'
##                EPOCH                              STOP
## DATE        START  #EPCHS    LR     END REASON                  EPOCH       ERROR   WGHT FILE    
##----------------------------------------------------------------------------------------------------------------
## 08-01-2019      0   2000   0.0001    end cancled                   82
## 09-01-2019      0   2000   0.0000001 end diskspace                406 
## 10-01-2019      0   2000   0.00001   end                 
## 12-01-2019      0   2000   0.00001   end                 
