#!/bin/bash
# Integration Check Script for nnU-Net with DynamicKiUNet

# echo "=========================================="
# echo "Step 1: Plan and Preprocess Dataset 004"
# echo "=========================================="
# nnUNetv2_plan_and_preprocess -d 004 --verify_dataset_integrity

# echo ""
# echo "=========================================="
# echo "Step 2: Train Default U-Net (3d_fullres)"
# echo "=========================================="
# nnUNetv2_train 004 3d_fullres 0

echo ""
echo "=========================================="
echo "Step 2: Clean previous training results"
echo "=========================================="
rm -rf /home/localssk23/nnn/datasets/nnUNet_results/Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres

echo ""
echo "=========================================="
echo "Step 3: Train DynamicKiUNet (2 epochs, minimal memory)"
echo "=========================================="
# Disable torch.compile to save memory
export nnUNet_compile="false"
nnUNetv2_train 004 3d_fullres 0 -tr kiunet_minimal

echo ""
echo "=========================================="
echo "Step 4: Predict with DynamicKiUNet"
echo "=========================================="
# Predict with DynamicKiUNet (config auto-detected from checkpoint)
nnUNetv2_predict -i /home/localssk23/nnn/datasets/raw/Dataset004_Hippocampus/imagesTs \
                 -o /home/localssk23/nnn/datasets/nnUNet_results/kiunet \
                 -d 004 \
                 -c 3d_fullres \
                 -f 0 \
                 -npp 5

echo ""
echo "=========================================="
echo "Integration Check Complete!"
echo "=========================================="
