call activate TF
python ../mrcnn/train_nshapes_mrcnn.py ^
    --epochs             2000          ^
    --steps_in_epoch       64          ^
    --val_steps            16          ^
    --last_epoch          268          ^
    --batch_size            1          ^
    --lr               0.0001          ^
    --mrcnn_logs_dir     train_mrcnn_newshapes  ^
    --mrcnn_model        last   ^
    --opt                adam   ^
    --sysout             file
::##  --new_log_folder  
call deactivate


::  ##----------------------------------------------------------------------------------------------------
::  ## Train only on COCO loaded classes loadAnns = 'active_only'
::  ##----------------------------------------------------------------------------------------------------
::  ## 19-12-2018  start  290, 500 epochs LR   0.0001  end early stop    @epoch 414  err 0.1367478 wght file 0314
::  ## 19-12-2018  start  414, 500 epochs LR  0.00001  end machine stop  @epoch 490  err  ?        wght file 0447      <--- train all layers 
::  ## 19-12-2018  start  490, 500 epochs LR  0.00001  end stop          @epoch 990  err 0.6111084 wght file 0541
::  ## 19-12-2018  start  990, 500 epochs LR 0.000001  end early         @epoch 1154 err 1.4481688 wght file 1034
::  ##   -12-2018  start 1154, 500 epochs LR  0.00001  end               @epoch      err           wght file  1
::  ##
::  ##----------------------------------------------------------------------------------------------------
::  ## Train only on COCO loaded classes loadAnns = 'active_only' - Training on all layers (res3+)
::  ##----------------------------------------------------------------------------------------------------
::  ## 20-12-2018  start    0, 100 epochs LR  0.0001  end stop           @epoch  100 err 0.48352  wght file 0066
::  ##   -12-2018  start  100, 1000epochs LR  0.0001  end early          @epoch  268 err 0.37132  wght file 0148 
::  ##   -12-2018  start    0,     epochs LR          end                @epoch   err    wght file  