nnUNetv2_plan_and_preprocess -d 004 --verify_dataset_integrity
nnUNetv2_train 004 3d_fullres 0
nnUNetv2_predict -i /home/localssk23/nnn/datasets/raw/Dataset004_Hippocampus/imagesTs -o /home/localssk23/nnn/datasets/nnUNet_results/ -d 004 -c 3d_fullres -f 0 -npp 5
