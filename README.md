This resporatory is attached to COV19D_2nd project. <br/>
The following lung segmentaion methods were implemented and their performance is measured using 'COVID-19 CT segmentation dataset'.
* The annotated CT images at http://medicalsegmentation.com/covid19/ were used.
* From the dataset; the "Image volumes (308 Mb)", and the "Lung masks (1 Mb)" were used to evaluate the segmentaiton performacne.
* The "Lung masks" were sliced in Z axial direction.
* The "Image Volumes" were sliced in Z axial direction, and the slices were then segmentaed using one of the proposed segmentation method. 
* The resulting volume slices were compared against the corresponding lung masks in terms of the dice Coefficient value. Average dice value and minimuim dice values were used to evaluate the segemtnation method.

# Region based segmentation using annotated public "COVID-19 CT segmentation dataset"
* The code 'Region-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb"
* The algorithm explores lung region based segmetnation.
* The Dice Coeffecient was used to measure the preformance of the segmetnation method. Average dice value over all slices= 0.89311, minimuim value = 0.74504

# Histogram threshold segmentation using annotated public "COVID-19 CT segmentation dataset"
* The code 'Threshold-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb"
* The algorithm explores lung otsu histogram based segmetnation.
* The Dice Coeffecient was used to measure the preformance of the segmetnation method. Average dice value over all slices = 0.89370, minimuim value = 0.75117


# K means clustering based segmentation using annotated public "COVID-19 CT segmentation dataset"
* The code 'K_means_Clustering_based_Segmentation_using_COVID_19_CT_segmentation_dataset.ipynb".
* The algorithm explores lung k-means clustering based segmentaion, where k=2.
* The Dice Coeffecient was used to measure the preformance of the segmetnation method. Average dice value over all slices = 0.89380, minimuim value = 0.75124
