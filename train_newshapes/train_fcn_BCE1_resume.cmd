## Training using BCE Method 1: Normal Loss optimzation (on all classes)

source activate TFG
python ../mrcnn/train_nshapes_fcn.py \
    --mrcnn_logs_dir    train_mrcnn_newshapes   \
    --fcn_logs_dir      train_fcn8L2_BCE        \
    --epochs               100       \
    --steps_in_epoch       128       \
    --val_steps             16       \
    --last_epoch             0       \
    --batch_size             8       \
    --lr                0.0001       \
    --fcn_arch          fcn8l2       \
    --fcn_losses        fcn_BCE_loss \
    --fcn_bce_loss_method    1       \
    --fcn_layers           all       \
    --mrcnn_model         last       \
    --fcn_model           init       \
    --opt                 adam       \
    --scale_factor           1       \
    --sysout               all       \
    --new_log_folder         

source deactivate


##--------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - All layers - w/ fcn_BCE_loss: Training on coco subset with loadAnns = 'active_only'
## With new Contextual Layer that assigns negative examples to individual classes (instead of moving then to class 0-BG)
##                EPOCH                              STOP
## DATE        START  #EPCHS    LR      END REASON                  EPOCH        ERROR      WGHT FILE    
##--------------------------------------------------------------------------------------------------------------
## 02-08-2019     0    1000   0.0001  
## 
## 
## 
##--------------------------------------------------------------------------------------------------------------
## Train only on COCO loaded classes loadAnns = 'active_only'
##--------------------------------------------------------------------------------------------------------------
## 19-12-2018  start  290, 500 epochs LR   0.0001  end early stop    @epoch 414  err 0.1367478 wght file 0314
## 19-12-2018  start  414, 500 epochs LR  0.00001  end machine stop  @epoch 490  err  ?        wght file 0447      <--- train all layers 
## 19-12-2018  start  490, 500 epochs LR  0.00001  end stop          @epoch 990  err 0.6111084 wght file 0541
## 19-12-2018  start  990, 500 epochs LR 0.000001  end early         @epoch 1154 err 1.4481688 wght file 1034
##   -12-2018  start 1154, 500 epochs LR  0.00001  end               @epoch      err           wght file  1
##--------------------------------------------------------------------------------------------------------------
