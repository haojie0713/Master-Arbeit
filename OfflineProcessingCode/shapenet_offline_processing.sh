#!/bin/bash
# ShapeNet Offline Processing Step1:  This shell script converts .obj files from ShapeNet to .pcd files in form of point cloud.

path=/home/haojie/Desktop/ShapeNetCore.v2

list_offsets=$(ls $path)

for offset in $list_offsets
do
    path1=$path/$offset
    list_objects=$(ls $path1)
    count=0
    mkdir $offset
    for object in $list_objects
    do
        path2=$path/$offset/$object/models/model_normalized.obj
        pcl_mesh_sampling $path2 $offset/$count.pcd -n_samples 6500 -leaf_size 0.001 -no_vis_result
        let "count+=1"
    done
done
