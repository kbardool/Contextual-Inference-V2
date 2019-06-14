source activate TFG
python ../mrcnn/build_coco_fcn_pred_structures.py \
       --mode                  test                \
       --batch_size            1                   \
       --dataset               coco                \
       --mrcnn_logs_dir        train_mrcnn_coco_subset "
       --fcn_logs_dir          train_fcn8L2_BCE_subset "
       --fcn_model             last                \
       --fcn_layer             all                 \
       --fcn_arch              fcn8L2              \
       --fcn_losses            fcn_BCE_loss        \
       --fcn_bce_loss_method   1                   \
       --sysout                screen              \
       --scale_factor          4
       --coco_classes          78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15
       
source deactivate       

## --mrcnn_model           /home/kbardool/models_newshapes2/train_mrcnn/mrcnn20190318T0000/mrcnn_0020.h5 \
