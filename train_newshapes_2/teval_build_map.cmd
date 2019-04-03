source activate TFG
python ../mrcnn/build_maps_structures.py \
       --mode                eval        \
       --evaluate_method     1           \
       --dataset             newshapes2
source deactivate