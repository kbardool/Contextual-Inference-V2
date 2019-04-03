source activate TFG
python ../mrcnn/build_nshapes_stats.py  \
    --dataset        newshapes2         \
    --batch_size     1                  \
    --mrcnn_logs_dir train_mrcnn        \
    --mrcnn_model    /home/kbardool/models_newshapes2/train_mrcnn/mrcnn20190318T0000/mrcnn_0020.h5 \
    --sysout         screen \
    --scale_factor   1      \
    --new_log_folder
    
source deactivate    
