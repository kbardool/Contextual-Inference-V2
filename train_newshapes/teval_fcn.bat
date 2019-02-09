call activate TF

python ../mrcnn/teval_nshapes_fcn.py ^
       --batch_size        1      ^
       --evaluate_method   1      ^
       --mrcnn_logs_dir    train_mrcnn_newshapes ^
       --fcn_logs_dir      train_fcn8L2_BCE2     ^
       --fcn_model         last   ^
       --fcn_layer         all    ^
       --fcn_arch          fcn8L2 ^
       --sysout            screen ^
       --scale_factor      1
       
 call deactivate