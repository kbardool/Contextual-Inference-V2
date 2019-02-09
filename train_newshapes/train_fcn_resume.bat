call activate TF
python ../mrcnn/train_nshapes_fcn.py ^
    --epochs              1000 ^
    --steps_in_epoch        16 ^
    --val_steps              8 ^
    --last_epoch          1620 ^
    --batch_size             1 ^
    --lr                0.0001 ^
    --mrcnn_logs_dir    train_mrcnn_newshapes   ^
    --fcn_logs_dir      train_fcn8_l2_newshapes ^
    --mrcnn_model       last   ^
    --fcn_model         last   ^
    --opt               adam   ^
    --fcn_arch          fcn8L2 ^
    --fcn_layers        block1+  ^
    --fcn_losses        fcn_BCE_loss ^
    --sysout            all   ^
    --scale_factor          1 ^
    --new_log_folder         

call deactivate


::----------------------------------------------------------------------------------------------------
:: Train only on COCO loaded classes loadAnns = 'active_only'
::----------------------------------------------------------------------------------------------------
:: 19-12-2018  start  290, 500 epochs LR   0.0001  end early stop    @epoch 414  err 0.1367478 wght file 0314
:: 19-12-2018  start  414, 500 epochs LR  0.00001  end machine stop  @epoch 490  err  ?        wght file 0447      <--- train all layers 
:: 19-12-2018  start  490, 500 epochs LR  0.00001  end stop          @epoch 990  err 0.6111084 wght file 0541
:: 19-12-2018  start  990, 500 epochs LR 0.000001  end early         @epoch 1154 err 1.4481688 wght file 1034
::   -12-2018  start 1154, 500 epochs LR  0.00001  end               @epoch      err           wght file  1
::
::--------------------------------------------------------------------------------------------------------------
:: Train FCN8L2 - All layers - w/ fcn_MSE_loss: Training on coco subset with loadAnns = 'active_only'
::                EPOCH                              STOP
:: DATE        START  #EPCHS    LR      END REASON                  EPOCH        ERROR      WGHT FILE    
::----------------------------------------------------------------------------------------------------------------
:: 20-12-2018     0     100   0.0001  end stop                       100        0.48352      0066
:: 25-12-2018   100    1000   0.0001  end early                      820        0.37132      0793 
:: 12-01-2019   820     800   0.0001  end normal                    1620        0.00623      1127
:: 16-01-2019  1620    1000   0.0001  end early stopping            2584        0.00526      2084
:: 
::    --epochs             2000          ^
::    --steps_in_epoch       64          ^
::    --val_steps            16          ^
::    --last_epoch          268          ^
::    --batch_size            1          ^
::    --lr               0.0001          ^
::    --mrcnn_logs_dir     train_mrcnn_newshapes  ^
::    --mrcnn_model        last   ^
::    --opt                adam   ^
::    --sysout             file
::##  --new_log_folder  
