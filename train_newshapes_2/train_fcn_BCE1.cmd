## Training using BCE Method 1: Normal Loss optimzation (on all classes)

source activate TFG
python ../mrcnn/train_nshapes_fcn.py \
    --dataset                  newshapes2   \
    --mrcnn_logs_dir           train_mrcnn  \
    --fcn_logs_dir             train_fcn8L2_BCE1 \
    --last_epoch               300       \
    --epochs                   300       \
    --steps_in_epoch           128       \
    --val_steps                 16       \
    --batch_size                 8       \
    --lr                       0.00001    \
    --fcn_arch                 fcn8l2    \
    --fcn_losses               fcn_BCE_loss \
    --fcn_bce_loss_method        1       \
    --fcn_layers               all       \
    --mrcnn_model              last      \
    --fcn_model                last      \
    --opt                      adam      \
    --scale_factor               1       \
    --sysout                    all      \
    --new_log_folder         

source deactivate


##--------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - All layers - w/ fcn_BCE_loss: Training on coco subset with loadAnns = 'active_only'
## With new Contextual Layer that assigns negative examples to individual classes (instead of moving then to class 0-BG)
##                EPOCH                              STOP                                    LAST  
## DATE        START  #EPCHS    LR      END REASON                  EPOCH        ERROR      WGHT FILE    
##--------------------------------------------------------------------------------------------------------------
## 02-08-2019     0    100   0.0001     
## 02-09-2019   100    400   0.0001    
## 
## 
##--------------------------------------------------------------------------------------------------------------
