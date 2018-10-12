##source /home/kbardool/anaconda3/bin/activate "TFG"
python train_coco_mrcnn.py           \
	--epochs 30   	            \
	--steps_in_epoch 10 	    \
	--last_epoch 20        \
	--batch_size 4         	\
	--lr 0.001 		            \
	--opt sgd 		            \
	--logs_dir train_mrcnn_coco     \
	--model last
##source /home/kbardool/anaconda3/bin/deactivate