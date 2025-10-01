#!/bin/bash
# Integration Check Script for nnU-Net with Custom Architectures (KiU-Net, UIU-Net)

# echo "=========================================="
# echo "Step 1: Plan and Preprocess Dataset 004"
# echo "=========================================="
# nnUNetv2_plan_and_preprocess -d 004 --verify_dataset_integrity

# echo ""
# echo "=========================================="
# echo "Step 2: Train Default U-Net (3d_fullres)"
# echo "=========================================="
# nnUNetv2_train 004 3d_fullres 0

# echo ""
# echo "=========================================="
# echo "Step 2: Clean previous training results"
# echo "=========================================="
# # Clean old checkpoints to ensure fresh training with updated architecture
# # Use nnUNet environment variables instead of hardcoded paths
# rm -rf "${nnUNet_results}/Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres"

# echo ""
# echo "=========================================="
# echo "Step 3: Train DynamicKiUNet (1 epoch, 50% features, batch_size=1)"
# echo "=========================================="
# # Use kiunet_minimal config - dual-branch arch needs reduced features for 24GB GPU
# # Disable torch.compile to save memory
# export nnUNet_compile="false"
# # Use cuda:1 device
# export CUDA_VISIBLE_DEVICES=1
# nnUNetv2_train 004 3d_fullres 0 -tr kiunet_minimal

# echo ""
# echo "=========================================="
# echo "Step 4: Predict with DynamicKiUNet"
# echo "=========================================="
# # Predict with DynamicKiUNet (config auto-detected from checkpoint)
# nnUNetv2_predict -i "${nnUNet_raw}/Dataset004_Hippocampus/imagesTs" \
#                  -o "${nnUNet_results}/kiunet" \
#                  -d 004 \
#                  -c 3d_fullres \
#                  -f 0 \
#                  -npp 5

echo ""
echo "=========================================="
echo "Step 5: Clean for UIU-Net training"
echo "=========================================="
# Clean KiU-Net results to prepare for UIU-Net
rm -rf "${nnUNet_results}/Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres"

echo ""
echo "=========================================="
echo "Step 6: Train DynamicUIUNet (1 epoch, 50% features, reduced RSU heights)"
echo "=========================================="
# Use uiunet_minimal config - nested U-Net structure is VERY memory intensive
# Disable torch.compile to save memory
export nnUNet_compile="false"
# Use cuda:1 device
export CUDA_VISIBLE_DEVICES=1
nnUNetv2_train 004 3d_fullres 0 -tr uiunet_minimal

echo ""
echo "=========================================="
echo "Step 7: Predict with DynamicUIUNet"
echo "=========================================="
# Predict with DynamicUIUNet (config auto-detected from checkpoint)
nnUNetv2_predict -i "${nnUNet_raw}/Dataset004_Hippocampus/imagesTs" \
                 -o "${nnUNet_results}/uiunet" \
                 -d 004 \
                 -c 3d_fullres \
                 -f 0 \
                 -npp 5

echo ""
echo "=========================================="
echo "Integration Check Complete!"
echo "=========================================="
