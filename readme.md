#Steps to remember for doing small stuff

1. No data aug `nnUNetv2_train 701 3d_lowres 0 --nnUNetTrainerNoDA.py --npz`
2. Use minimal number of images
3. 3d_lowres
4. Batch_size = 1


`rsync -av --exclude='validation' --exclude='*.pth' --exclude='*.json' --exclude='*.txt' sskundu@h1:/nfs/home/sskundu/nnunet/nnUNet_results/Dataset2024_Mets ./ ` #Get the outputs locally  nicely
