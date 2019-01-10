source activate TFG
python ./mrcnn/build_heatmap_npz.py    \
    --output_dir       coco2014_heatmaps/train2014 \
    --dataset          train val35k \
    --iterations            3  \
    --batch_size            2  \
    --start_from            0  \
    --sysout           screen

#    --epochs               1  \
#    --steps_in_epoch       12   \
#    --last_epoch            0   \
#    --lr                  0.01   \
#    --val_steps             4   \
#    --mrcnn_logs_dir   train_mrcnn_coco \
#    --fcn_logs_dir     train_fcn_coco \
#    --new_log_folder  
source deactivate
