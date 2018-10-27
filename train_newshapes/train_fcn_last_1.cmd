source activate TFG
python train_nshapes_fcn.py         \
  --epochs          100   \
  --steps_in_epoch  400   \
  --last_epoch     4524   \
  --batch_size       32   \
  --val_steps        16   \
  --lr           1.0e-8   \
  --opt          rmsprop  \
  --logs_dir     train_fcn_rmsprop  \
  --model         /home/kbardool/models/train_mrcnn/shapes20180621T1554/mask_rcnn_shapes_1119.h5 \
  --fcn_model    last
    

source deactivate

## continue SGD with rmsprop
##              steps/epoch      batch sz       val steps    
## 475 -> 840:  32, 25, 5  
## 841 -> 900:   5,  8, 2
## 900 -> 990
## 990 -> 1900 -- changed Reduce on LR parms 
## 1900 -> 2900
## epochs          epochs    steps in epoch   batch sz     val_steps    LR           opt
#  3358 -- 4024    5000 (f)        12             8            2        1.0e-6 (?)   rmsprop
#  4024 -- 4524     500 (s)        12             2            1        1.0e-8       ""
#  4524 --          500            12            16            4        1.0e-8       rmsprop
#
#
#
#    --epochs         1000   \
#    --steps_in_epoch    5   \
#    --last_epoch     2900   \
#    --batch_size        8   \
#    --lr              1.0   \
#    --val_steps         2   \
#    --opt           rmsprop   \
#    --logs_dir      train_fcn_rmsprop \
#  --model    /esat/tiger/joramas/mscStudentsData/kbardool/models/train_mrcnn/mrcnn20180621T1554/mask_rcnn_shapes_1119.h5 \
#    --model         /home/kbardool/models/train_mrcnn/shapes20180621T1554/mask_rcnn_shapes_1119.h5 \
#    --fcn_model     /home/kbardool/models/train_fcn_rmsprop/fcn20180730T0813/fcn_1909.h5
