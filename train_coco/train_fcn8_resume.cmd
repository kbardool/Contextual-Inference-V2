source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs              500   \
    --steps_in_epoch       64   \
    --last_epoch          732   \
    --batch_size            1   \
    --lr               0.00001   \
    --val_steps            16   \
    --fcn_arch           fcn8   \
    --fcn_layers          all   \
    --fcn_losses         fcn_BCE_loss \
    --mrcnn_logs_dir     train_mrcnn_coco \
    --fcn_logs_dir       train_fcn8_bce \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --sysout             file
##  --new_log_folder     \
    
source deactivate

##  --new_log_folder  
##    --lr               0.0001   \
##----------------------------------------------------------------------------------------------------
## running over fcn_BCE_loss: Folder  fcn20181205T0000
##----------------------------------------------------------------------------------------------------
## 05-12-2018  start   0, 100 epochs LR 0.001  end normal       @epoch  100 err 0.01181   wght file ????
## 05-12-2018  start 100, 400 epochs LR 0.0001 end machine stop @epoch  245 err 0.0073236 wght file 0135
## 06-12-2018  start 245, 400 epochs LR 0.0001 end disk error   @epoch  267 err 0.0090296 wght file 0257
## 08-12-2018  start 434, 400 epochs LR 0.0001 end machine stop @epoch  732 err 0.0082694 wght file 0671
## 15-12-2018  start 732, 500 epochs LR 0.0001 end early stop   @epoch 1014 err 0.0053342 wght file 0864
##
##----------------------------------------------------------------------------------------------------
## running on coco subset : Folder fcn20181128T0000
##----------------------------------------------------------------------------------------------------
##  28-11-2018  start 0100 4000 epochs - ended @ 473 epochs earlystopping 0.0000375 weight file 0274
##  29-11-2018  start 0473 4000 epochs - ended @ 730 epochs earlystopping 0.0000445 weight file 0580
##  29-11-2018  start 0730 4000 epochs - ended @ 900 epochs earlystopping 0.0000512 weight file 0750
##  02-12-2018  start 0900 40   epochs - eded  @ 940                   ~  0.00008xx weight file 0908  
##
##----------------------------------------------------------------------------------------------------
## Full COCO  folder:              Error: MSE ??
##----------------------------------------------------------------------------------------------------
##  11-11-2018   last epoch 786
##  13-11-2018   start 786  1000 epochs ended Early Stopping at 1100 error 0.0000440
##  15-11-2018   start 1100 1000 epochs ended due to machine stop error ~ 0.0000456
##  16-11-2018   start 1674 1000 start  with LR 0.001  ended 2036 early stopping
##  17-11-2018   start 2036 4000 epochs with LR 0.0001
##                          2051 Error : 0.0000496
##                          2298 Error : 0.0000398  ended epoch 3047 machine shut down
##  18-11-2018   start 3047 2000 epochs
##



