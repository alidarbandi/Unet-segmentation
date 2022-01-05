# Unet-segmentation
semantic segmentation of electron microscope images (TEM, and FIB-SEM) with single class UNet model 

# Step by step manual
1. Start with pre-processing the EM images. I have separate codes for pre-processing. This includes processes for image binning, cropping to ROI, and noise filteration. 
2. Prepare image masks. These are the single class segmented binary images for training. You can use manual segmentation/tracing or use software such as Dragonfly to prepare the masks. Masks and training images must have same size and pixel coordinates. 
3. Prepare a validation data set. This is similar to training data set a smaller volume to assess the model accuracy. 
4. Carvana_data.py includes the pytorch class that handles data reading. It doesn't need modification.
5. utils.py includes functions for pytorch dataloader, data augumentation, and accuracy checker. Edit the directory path for saving your files inside the save_pre fucntion
6. unet_tutorial.py is the main UNet model based on Ronneberger paper https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/ . You can see the architecture in the attached PNG image.
Unet_training.py is the training script. Download all the python files from the repository and put them in your working directory. Edit the file paths of image_dir , mask_dir for training data set and val_image_dir , val_mask_dir for validation data set. Also edit the path for saving the trained model. The rest of the files doesnt need any modification. 
Install all the required python packages, Pytorch, Albumentations, numpy, PIL. 
Start the training by executing unet_training.py. The script will return the valication accuracy and dice score for each epoch.
Adjust the hyperparameters to optimize the training accuracy. i.e. Epoch, learning rate, batch size. 
