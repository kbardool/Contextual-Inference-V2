source activate TFG
python train_coco_fcn.py  \
  --epochs          100   \
  --steps_in_epoch  400   \
  --last_epoch        0   \
  --batch_size       32   \
  --val_steps        16   \
  --lr             0.01   \
  --opt          rmsprop  \
  --logs_dir     coco_fcn \
  --model        coco     \
  --fcn_model    last
source deactivate
