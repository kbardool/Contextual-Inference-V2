source activate TFG
python ../mrcnn/train_coco_fcn.py    \
    --epochs              413   \
    --steps_in_epoch       64   \
    --last_epoch         1587   \
    --batch_size            1   \
    --lr              0.000001  \
    --val_steps            16   \
    --fcn_arch           fcn8l2 \
    --fcn_layers         all    \
    --fcn_losses         fcn_MSE_loss \
    --mrcnn_logs_dir     train_mrcnn_coco_subset \
    --fcn_logs_dir       train_fcn8L2_MSE_subset \
    --mrcnn_model        last   \
    --fcn_model          last   \
    --opt                adam   \
    --coco_classes       78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 \
    --sysout             ALL   \
    --new_log_folder
    
source deactivate

#
# --coco_classes:
#  appliance: 78 - 82   kitchen: 44 - 51   sports: 34 - 43       indoor: 10 -15
#
##--------------------------------------------------------------------------------------------------------------
## Train FCN8L2 - All layers - w/ fcn_MSE_loss: Training on coco subset with loadAnns = 'active_only'
##                EPOCH                              STOP
## DATE        START  #EPCHS    LR       END REASON              END EPOCH        ERROR      WGHT FILE    
##--------------------------------------------------------------------------------------------------------------
## 08-01-2019      0   2000     0.0001   cancled                        82   
## -------------------------------------------------------------------------------------------------------------                                                                         
## 09-01-2019      0   2000  0.0000001   diskspace                     406   
## -------------------------------------------------------------------------------------------------------------                                                                         
##                                                                           
## 10-01-2019      0   2000    0.00001   diskspace                      41      0.0050676    0041                
## 10-01-2019     41   2000     0.0001   machine shutdown              407    0.000013505    0167  
## 10-01-2019    407   2000   0.000001   machine shutdown              864    0.000012116    0670
## -------------------------------------------------------------------------------------------------------------                                                                         
## 12-01-2019      0   2000    0.00001   diskspace                     116     0.0011477     0116   
## 13-01-2019    116   2000    0.00001   machine shutdown              175     0.00041525    0175   
## 13-01-2019    175   2000    0.00001   machine shutdown              615     0.0000180     0542                            
## 14-01-2019    615   1385    0.00001   machine shutdown             1014     0.0000171     0690
## 15-01-2019   1014    986    0.00001   machine shutdown             1450     0.0000180     1228
## 17-01-2019   1450    550   0.000001   machine shutdown             1493     0.0000210     1492                             
## 18-01-2019   1493    508   0.000001   machine shutdown             1587     0.0000190     1568
## 19-01-2019   1587    413   0.000001                                2000     0.0000167     1806 
## 




