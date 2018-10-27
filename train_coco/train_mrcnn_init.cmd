source activate TFG
python train_coco_mrcnn.py  \
        --epochs         10 \
        --steps_in_epoch 10 \
        --last_epoch      0 \
        --batch_size      2 \
        --lr          0.001 \
        --logs_dir train_mrcnn_coco  \
        --model coco         
source deactivate