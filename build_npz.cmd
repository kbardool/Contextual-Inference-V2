source activate TFG
python ./mrcnn/build_heatmap_npz.py    \
    --model              last   \
    --output_dir       coco2014_heatmaps/train_heatmaps \
    --iterations          125   \
    --batch_size            2   \
    --start_from          150   \
    --sysout             screen

#    --epochs               1  \
#    --steps_in_epoch       12   \
#    --last_epoch            0   \
#    --lr                  0.01   \
#    --val_steps             4   \
#    --mrcnn_logs_dir   train_mrcnn_coco \
#    --fcn_logs_dir     train_fcn_coco \
#    --new_log_folder  
source deactivate
