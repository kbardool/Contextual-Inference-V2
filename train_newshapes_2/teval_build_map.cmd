source activate TFG
python ../mrcnn/build_map_structures.py \
       --mode                eval        \
       --evaluate_method     2           \
       --dataset             newshapes2
source deactivate