source activate TFG
python ./mrcnn/build_heatmap_npz.py    \
    --model              last   \
    --output_dir       coco2014_heatmaps/train_heatmaps \
    --iterations          100   \
    --batch_size            2   \
    --start_from         1000   \
    --sysout             screen

source deactivate
