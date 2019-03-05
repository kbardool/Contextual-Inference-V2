source activate TFG
python train_nshapes_mrcnn.py           \
    --epochs             1000  \
    --steps_in_epoch      128  \
    --last_epoch            0  \
    --batch_size           16  \
    --lr                0.001  \
    --mrcnn_logs_dir     train_mrcnn_newshapes  \
    --fcn_logs_dir       train_fcn_newshapes    \
    --mrcnn_model        init  \
    --val_steps            32  \
    --opt                adam  \
    --sysout              all  \
    --new_log_folder  
##	--model "/esat/tiger/joramas/mscStudentsData/kbardool/models/train_mrcnn/shapes20180621T1554/mask_rcnn_shapes_1119.h5"

source deactivate