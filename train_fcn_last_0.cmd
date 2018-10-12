python train_nshapes_fcn.py         \
    --epochs         1000   \
    --steps_in_epoch    5   \
    --last_epoch     2900   \
    --batch_size        8   \
    --lr              1.0   \
    --val_steps         2   \
    --opt           rmsprop   \
    --logs_dir      train_fcn_rmsprop \
    --model         /home/kbardool/models/train_mrcnn/shapes20180621T1554/mask_rcnn_shapes_1119.h5 \
    --fcn_model     /home/kbardool/models/train_fcn_rmsprop/fcn20180730T0813/fcn_1909.h5
    
## continue SGD with rmsprop
##              steps/epoch      batch sz       val steps    
## 475 -> 840:  32, 25, 5  
## 841 -> 900:   5,  8, 2
## 900 -> 990
## 990 -> 1900 -- changed Reduce on LR parms 
## 1900 -> 2900
