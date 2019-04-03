call source activate TFG
python ../mrcnn/train_nshapes_mrcnn.py  ^
    --epochs              100  ^
    --steps_in_epoch      128  ^
    --last_epoch            0  ^
    --batch_size            1  ^
    --lr                0.001  ^
    --mrcnn_logs_dir     train_mrcnn_newshapes  ^
    --mrcnn_layers       res3+ ^
    --mrcnn_model        last  ^
    --val_steps            16  ^
    --opt                adam  ^
    --sysout             file  ^
    --new_log_folder  

call source deactivate    
