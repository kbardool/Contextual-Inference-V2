python train_nshapes_mrcnn.py           \
	--epochs 1000 	            \
	--steps_in_epoch 128 	    \
	--last_epoch 1198 	        \
	--batch_size 32         	\
	--lr 0.01 		            \
	--opt sgd 		            \
	--logs_dir train_mrcnn      \
	--model "/esat/tiger/joramas/mscStudentsData/kbardool/models/train_mrcnn/shapes20180621T1554/mask_rcnn_shapes_1119.h5"
