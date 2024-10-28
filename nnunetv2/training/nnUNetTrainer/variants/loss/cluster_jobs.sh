DOCKER_IMAGE="aicregistry:5000/sskundu:lesionv2"
USERNAME="sskundu"
MOUNT_DIRECTORY="/nfs:/nfs"

raw="/nfs/home/sskundu/nnUNet/nnUNet_raw"
preprocess="/nfs/home/sskundu/nnUNet/nnUNet_preprocess"
result="/nfs/home/sskundu/nnUNet/nnUNet_results"

for fold in {0..2}; do

    # Baseline (Dice + CE)
    JOB_NAME="brats2024-baseline-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold --npz
    
    ##########
    # Single #
    ##########

    # Dice
    JOB_NAME="brats2024-dice-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerDiceLoss --npz
    
    # CE
    JOB_NAME="brats2024-ce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerCELoss --npz
    
    # Tversky
    JOB_NAME="brats2024-tversky-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerTverskyLoss --npz
    
    # Topk
    JOB_NAME="brats2024-topk-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerTopkLoss --npz
    
    # RCE
    JOB_NAME="brats2024-rce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerRCELoss --npz
    
    # bDice
    JOB_NAME="brats2024-bdice-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbDiceLoss --npz
    
    # bTversky
    JOB_NAME="brats2024-btversky-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbTverskyLoss --npz
    
    ############
    # Compound #
    ############

    # Dice + RCE
    JOB_NAME="brats2024-dice-rce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerDice_RCELoss --npz
    
    # Dice + Topk
    JOB_NAME="brats2024-dice-topk-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerDice_TopkLoss --npz
    
    # Tversky + CE
    JOB_NAME="brats2024-tversky-ce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerTversky_CELoss --npz
    
    # Tversky + RCE
    JOB_NAME="brats2024-tversky-rce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerTversky_RCELoss --npz
    
    # Tversky + Topk
    JOB_NAME="brats2024-tversky-topk-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerTversky_TopKLoss --npz
    
    ########
    # Blob #
    ########

    # bDice + CE
    JOB_NAME="brats2024-bdice-ce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbDice_CELoss --npz
    
    # bDice + RCE
    JOB_NAME="brats2024-bdice-rce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbDice_RCELoss --npz
    
    # bDice + Topk
    JOB_NAME="brats2024-bdice-topk-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbDice_TopKLoss --npz
    
    # bTversky + CE
    JOB_NAME="brats2024-btversky-ce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbTversky_CELoss --npz
    
    # bTversky + RCE
    JOB_NAME="brats2024-btversky-rce-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbTversky_RCELoss --npz
    
    # bTversky + Topk
    JOB_NAME="brats2024-btversky-topk-$fold"
    runai submit --name $JOB_NAME \
        --image $DOCKER_IMAGE \
        --run-as-user \
        --project $USERNAME \
        -v $MOUNT_DIRECTORY \
        --backoff-limit 0 \
        --gpu 1 \
        --host-ipc \
        -e "ENV_VAR_NAME1=$raw" \
        -e "ENV_VAR_NAME2=$preprocess" \
        -e "ENV_VAR_NAME3=$result" \
        --command -- nnUNetv2_train 2024 --c 3d_fullres $fold -tr nnUNetTrainerbTversky_TopKLoss --npz