source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs               900       \
    --steps_in_epoch        32       \
    --val_steps              8       \
    --last_epoch          3674       \
    --batch_size             1       \
    --lr               0.00001       \
    --fcn_arch          fcn8l2       \
    --fcn_losses        fcn_BCE_loss \
    --fcn_layers        all          \    
    --mrcnn_logs_dir    train_mrcnn_coco_subset \
    --fcn_logs_dir      train_fcn8L2_BCE_subset \
    --mrcnn_model       last         \
    --fcn_model         last         \
    --opt               adam         \
    --coco_classes      78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --sysout            ALL   \
    --new_log_folder
    
source deactivate

#
# --coco_classes:
#  appliance: 78 - 82   kitchen: 44 - 51   sports: 34 - 43       indoor: 10 -15
##----------------------------------------------------------------------------------------------------------------
##                EPOCH                              STOP
## DATE        START  #EPCHS    LR     END REASON             END EPOCH      ERROR   WGHT FILE    
##----------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - all layers - w/ fcn_BCE_loss: Folder  weight decay: 1.0e-6 (I think loadAnns = 'allclasses)
##----------------------------------------------------------------------------------------------------------------
## 16-12-2018     0     500   0.0001   machine shutdown             218    0.0072799  0154  folder\fcn20181216T0000
## 17-12-2018   218     500  0.00005   early stop                   449    0.0058245  0300
##
##----------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - All layers - w/ fcn_BCE_loss: Training on coco subset with loadAnns = 'active_only'
##----------------------------------------------------------------------------------------------------------------
## 21-12-2018      0   2000   0.0001   end disk space                25    0.0131407  0010   (batch 64, val 8)
## 21-12-2018     25   2000   0.0001   end machine shutdown         583    0.0037354  0547   (batch 64, val 8)
## 22-12-2018    583   2000   0.0001   end machine shutdown        1064    0.0027937  0950   (batch 64, val 8)



## ------- Restart from epoch 1064 wght file 0950 LR 0.0001 -----
## 30-12-2018   1064   2000   0.000001 not sure why                1500    0.0030359  1108   (batch 64, val 8)
## 20-01-2019   1594    506   0.0001   end machine shutdown        2019    0.0035939  1679   (batch 64, val 16) gap in epochs 1500<>1594 
## 20-01-2019   2019    781   0.0001   normal completion           2799    0.0023078  2330   (batch 32, val 8)
## 22-01-2019   2799   1000   0.00001  machine shutdown            3674    0.0030780  3348   (batch 32, val 8)
## 23-01-2019   3674    926   0.00001                              4516    0.0038953  4345   (batch 32, val 8)
##
## 
## 
## 
## 
## 
## 
## 
## 
