# RSNA 2022 Cervical Spine Fracture Detection 7th solution
Fracture Detection ， Fracture Segmentation ， nn-UNet  

## Overview
This solution is based on [nnUnet](https://github.com/MIC-DKFZ/batchgenerators) and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
![image](https://github.com/LSL000UD/RSNA2022-7th-Place/blob/main/overview.png)

In our solution,  3D segmentation methods are utilized for fracture detection task. Since host do not provide segmentation label for fracture region, we use data-augmentations and bounding box GT to generate Pseudo segmentation masks. Our final framework consist of 3 stages:

- Stage 1: Segment C1-C7 using 3D-UNet

- Stage 2: Segment bone fracture region using 3D-UNet

- Stage 3: Predict final score using outputs from Stage 1 and Stage 2



## Requirements
- torch==1.11.0+cu115
- Python 3.7+
- At least 24 GB GPU memory


## Training

Training codes are directly modified on nn-UNet, so it may not be well organized.

1. Stage 1
   	
	- Download [Competition Data](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/data), [TotalSegmentator Data(CC-BY-4.0)](https://zenodo.org/record/6802614#.Y2nkrHYzZPY) and [Verse2020 Data(CC-BY-4.0)](https://github.com/anjany/verse)
	- Speicfy all path in path.py
	- Convert competition data to NIFTI image
	   -  cd /DatasetConversion/
	   -  python step0_trans_DICOM_to_Nii.py
	   -  python step1_trans_segmentation_to_nii.py
	   -  python step2_trans_bbox_to_nii.py
	   -  python step3_trans_public_datasets_to_nii.py
	   -  python step4_split_train_val.py
	
	- Follow nnUnet workflow to train the model and get prediction results of stage1 
		- cd /Training/Task_101_VertebralLocation_GeneratePseudoLabel/
		- python nnUNet_prepare_raw_data.py
		- python nnUNet_plan_and_preprocess.py
		- python nnUNet_change_plan.py
		- python nnUNet_run_training.py
		- python inference.py

2. Stage 2 
 (This stage using Pseudo labeling techniques. so it need 3 iterations to get final label to train the final model)
	- Follow nnUnet workflow to train the model and get prediction results of **stage2.1**
		- cd /Training/Task_201_FractureDetection_GeneratePseudoLabel/
		- python nnUNet_prepare_raw_data.py
		- python nnUNet_plan_and_preprocess.py
		- python nnUNet_change_plan.py
		- python nnUNet_run_training.py
		- python inference.py
<br/><br/><br/>
	- Follow nnUnet workflow to train the model and get prediction results of **stage2.2**
		- cd /Training/Task_202_FractureDetection/
		- python nnUNet_prepare_raw_data.py
		- python nnUNet_plan_and_preprocess.py
		- python nnUNet_change_plan.py
		- python nnUNet_run_training.py
		- python inference.py
<br/><br/><br/>
	- Follow nnUnet workflow to train the model and get prediction results of **stage2.3**
		- cd /Training/Task_203_FractureDetection_Real5Fold/
		- python nnUNet_prepare_raw_data.py
		- python nnUNet_plan_and_preprocess.py
		- python nnUNet_change_plan.py
		- python nnUNet_run_training.py
		- python final_inference_CV.py
		- python final_inference_remain.py

3. Stage 3
	- Training stage 3 model
	   -  cd /Task_301_PostProcessing_Overall/
	   -  python prepare_raw_data.py
	   -  python train.py

	
## Testing

After training, you can use this notebook for inference https://www.kaggle.com/code/lsl000ud/rsna2022-7th-place-inference

## Acknowledgement
-Thank [nnUnet](https://github.com/MIC-DKFZ/batchgenerators), [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
and  RSNA 2022 Organizers
	-https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/overview

