#!/bin/bash
# Integration Check Script for nnU-Net with Custom Architectures (KiU-Net, UIU-Net)

set -e  # Exit on error

# Default values
DATASET=004
GPU=1
RUN_PREPROCESS=false
RUN_BASELINE=false
RUN_KIUNET=false
RUN_UIUNET=false
CLEAN=true

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help message
show_help() {
    cat << EOF
Usage: ${0##*/} [OPTIONS]

Integration test script for nnU-Net custom architectures.

OPTIONS:
    -h, --help          Show this help message
    -d, --dataset N     Dataset ID (default: 004)
    -g, --gpu N         GPU device ID (default: 1)
    --all               Run all steps (preprocess + baseline + kiunet + uiunet)
    --preprocess        Run preprocessing step
    --baseline          Train baseline U-Net
    --kiunet            Train and predict with KiU-Net
    --uiunet            Train and predict with UIU-Net
    --no-clean          Don't clean previous results before training

EXAMPLES:
    # Run only UIU-Net test
    ${0##*/} --uiunet

    # Run KiU-Net and UIU-Net
    ${0##*/} --kiunet --uiunet

    # Run everything including preprocessing
    ${0##*/} --all

    # Run UIU-Net on GPU 0 without cleaning previous results
    ${0##*/} --uiunet --gpu 0 --no-clean

    # Run baseline and UIU-Net on dataset 005
    ${0##*/} --baseline --uiunet -d 005

EOF
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No options specified. Use --help for usage information.${NC}"
    echo -e "${YELLOW}Running UIU-Net by default...${NC}\n"
    RUN_UIUNET=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        --all)
            RUN_PREPROCESS=true
            RUN_BASELINE=true
            RUN_KIUNET=true
            RUN_UIUNET=true
            shift
            ;;
        --preprocess)
            RUN_PREPROCESS=true
            shift
            ;;
        --baseline)
            RUN_BASELINE=true
            shift
            ;;
        --kiunet)
            RUN_KIUNET=true
            shift
            ;;
        --uiunet)
            RUN_UIUNET=true
            shift
            ;;
        --no-clean)
            CLEAN=false
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Export environment variables
export nnUNet_compile="false"
export CUDA_VISIBLE_DEVICES=$GPU

echo -e "${GREEN}=========================================="
echo "nnU-Net Integration Check"
echo "=========================================="
echo -e "Dataset: ${BLUE}$DATASET${NC}"
echo -e "GPU: ${BLUE}$GPU${NC}"
echo -e "Preprocess: ${BLUE}$RUN_PREPROCESS${NC}"
echo -e "Baseline U-Net: ${BLUE}$RUN_BASELINE${NC}"
echo -e "KiU-Net: ${BLUE}$RUN_KIUNET${NC}"
echo -e "UIU-Net: ${BLUE}$RUN_UIUNET${NC}"
echo -e "Clean previous results: ${BLUE}$CLEAN${NC}"
echo -e "${GREEN}==========================================${NC}\n"

# Step 1: Preprocessing
if [ "$RUN_PREPROCESS" = true ]; then
    echo -e "${GREEN}=========================================="
    echo "Step 1: Plan and Preprocess Dataset $DATASET"
    echo -e "==========================================${NC}"
    nnUNetv2_plan_and_preprocess -d $DATASET --verify_dataset_integrity
    echo ""
fi

# Step 2: Baseline U-Net
if [ "$RUN_BASELINE" = true ]; then
    if [ "$CLEAN" = true ]; then
        echo -e "${GREEN}=========================================="
        echo "Cleaning previous baseline results"
        echo -e "==========================================${NC}"
        rm -rf "${nnUNet_results}/Dataset$(printf '%03d' $DATASET)_*/nnUNetTrainer__nnUNetPlans__3d_fullres"
        echo ""
    fi

    echo -e "${GREEN}=========================================="
    echo "Step 2: Train Baseline U-Net"
    echo -e "==========================================${NC}"
    nnUNetv2_train $DATASET 3d_fullres 0
    echo ""
fi

# Step 3: KiU-Net
if [ "$RUN_KIUNET" = true ]; then
    if [ "$CLEAN" = true ]; then
        echo -e "${GREEN}=========================================="
        echo "Cleaning previous KiU-Net results"
        echo -e "==========================================${NC}"
        rm -rf "${nnUNet_results}/Dataset$(printf '%03d' $DATASET)_*/nnUNetTrainer__nnUNetPlans__3d_fullres"
        echo ""
    fi

    echo -e "${GREEN}=========================================="
    echo "Step 3: Train KiU-Net (1 epoch, 50% features)"
    echo -e "==========================================${NC}"
    nnUNetv2_train $DATASET 3d_fullres 0 -tr kiunet_minimal
    echo ""

    echo -e "${GREEN}=========================================="
    echo "Step 4: Predict with KiU-Net"
    echo -e "==========================================${NC}"
    # Find the actual dataset directory name
    DATASET_DIR=$(ls -d "${nnUNet_raw}"/Dataset$(printf '%03d' $DATASET)_* 2>/dev/null | head -1)
    if [ -n "$DATASET_DIR" ] && [ -d "$DATASET_DIR/imagesTs" ]; then
        nnUNetv2_predict -i "$DATASET_DIR/imagesTs" \
                         -o "${nnUNet_results}/kiunet_predictions" \
                         -d $DATASET \
                         -c 3d_fullres \
                         -f 0 \
                         -npp 5
    else
        echo -e "${YELLOW}Warning: Test images directory not found, skipping prediction${NC}"
    fi
    echo ""
fi

# Step 4: UIU-Net
if [ "$RUN_UIUNET" = true ]; then
    if [ "$CLEAN" = true ]; then
        echo -e "${GREEN}=========================================="
        echo "Cleaning previous UIU-Net results"
        echo -e "==========================================${NC}"
        rm -rf "${nnUNet_results}/Dataset$(printf '%03d' $DATASET)_*/nnUNetTrainer__nnUNetPlans__3d_fullres"
        echo ""
    fi

    echo -e "${GREEN}=========================================="
    echo "Step 5: Train UIU-Net (1 epoch, 50% features, reduced RSU heights)"
    echo -e "==========================================${NC}"
    nnUNetv2_train $DATASET 3d_fullres 0 -tr uiunet_minimal
    echo ""

    echo -e "${GREEN}=========================================="
    echo "Step 6: Predict with UIU-Net"
    echo -e "==========================================${NC}"
    # Find the actual dataset directory name
    DATASET_DIR=$(ls -d "${nnUNet_raw}"/Dataset$(printf '%03d' $DATASET)_* 2>/dev/null | head -1)
    if [ -n "$DATASET_DIR" ] && [ -d "$DATASET_DIR/imagesTs" ]; then
        nnUNetv2_predict -i "$DATASET_DIR/imagesTs" \
                         -o "${nnUNet_results}/uiunet_predictions" \
                         -d $DATASET \
                         -c 3d_fullres \
                         -f 0 \
                         -npp 5
    else
        echo -e "${YELLOW}Warning: Test images directory not found, skipping prediction${NC}"
    fi
    echo ""
fi

echo -e "${GREEN}=========================================="
echo "Integration Check Complete!"
echo -e "==========================================${NC}"
