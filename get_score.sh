#!/bin/bash
#This file is used for copying landmarks.txt to "_result_set", then runing the application to extract ROI/ vein images in "_result_set" and compute the score.
#Ensure that the 'landmarks.txt' files are already located within the NTUST_HS_Testset (test dataset) folder.
#NOTE: modify MODELS in FOR LOOP, and modify 'model_lists' at the beginning of compute_score.ipynb
loop_string=200

source="$(pwd)/source/_database/NTUST_HS_Testset/train_shadowfree/landmarks.txt"
destination="$(pwd)/source/_result_set/groudtruth"
eval "cp ${source} ${destination}"

source="$(pwd)/source/_database/NTUST_HS_Testset/train_shadowfull/landmarks.txt"
destination="$(pwd)/source/_result_set/original"
eval "cp ${source} ${destination}"
    
for MODEL in "STGAN" "SIDSTGAN" "SIDPAMIwISTGAN"
do
    for VARIABLE in "latest" #$(seq 10 10 $loop_string)
    do
        destination="$(pwd)/source/_result_set/shadowfree_${MODEL}_rawsynthetic_"
        dir="${destination}${VARIABLE}"
        eval "cp ${source} ${dir}"  

        destination="$(pwd)/source/_result_set/shadowfree_${MODEL}_rawsynthetic_HandSeg_"
        dir="${destination}${VARIABLE}"
        eval "cp ${source} ${dir}"  
    
        echo "Model: ${MODEL}, checkpoints: ${VARIABLE}"
    done
done

eval "$(pwd)/Extract_Vein_By_Trung/Extract_Vein_By_Trung/bin/Release/net6.0-windows/Extract_Vein_By_Trung.exe"
eval "jupyter notebook $(pwd)/compute_score.ipynb" #Open .sh file by in anaconda environment
$SHELL 
