* This repository is segemtnation part of the pipeline proposed [here](https://github.com/IDU-CVLab/COV19D_3rd)
* Different segmentaion methods were compared and their performances were recorded and measured using dice similarities. The results were evaluated using the publicly available [COVID-19 CT segmentation dataset](http://medicalsegmentation.com/covid19/).
* From the dataset; the "Image volumes (308 Mb)", and the "Lung masks (1 Mb)" were used for evaluation.
* The "Image Volumes" were sliced in Z axial direction, and the slices were then segmentaed using one of the proposed segmentation methods.
* Training set includes all t2 to t8 images and correspoding masks; i.e. 745 annotated images and 745 annotated masks.
* Test set includes all t0 and t1 images and corresponding masks; i.e. 84 annotated images and 84 annotated masks.  
* The resulting volume slices were compared against the corresponding lung masks in terms of the dice Coefficient value for two classes. Average dice value and minimuim dice value over all lung mask slices were used to evaluate the segemtnation method. <br/>

The codes are:
1. 'Region-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb'
2. 'Threshold-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb'
3. 'K_means_Clustering_based_Segmentation_using_COVID_19_CT_segmentation_dataset.ipynb'
4.  'UNet_model.py'
