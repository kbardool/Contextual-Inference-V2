python train_nshapes_fcn.py     \
    --epochs          300   \
    --steps_in_epoch  128   \
    --last_epoch        0   \
    --batch_size        8   \
    --lr            0.001   \
    --val_steps         5   \
    --opt           kevin   \
    --logs_dir      train_fcn_alt  \
    --model         /home/kbardool/models/newshape_mrcnn/shapes20180621T1554/mask_rcnn_shapes_1119.h5 \
    --fcn_model     init
    
    
## python train_fcn_alt.py  --epochs 100 --steps_in_epoch 128  --last_epoch 0  --batch_size 16 --lr 0.01 --logs_dir train_fcn --model last --fcn_model init
