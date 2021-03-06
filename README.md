This resporatory is part of the [COV19D_2nd project](https://github.com/IDU-CVLab/COV19D_2nd). <br/>
* Lung segmentaion methods were used and their performances were measured in terms of dice similarity. The results were evaluated using the publicly available 'COVID-19 CT segmentation dataset' at http://medicalsegmentation.com/covid19/.
* From the dataset; the "Image volumes (308 Mb)", and the "Lung masks (1 Mb)" were used to evaluate the segmentaiton performacne.
* The "Lung masks" were sliced in Z axial direction.
* The "Image Volumes" were sliced in Z axial direction, and the slices were then segmentaed using one of the proposed segmentation methods. 
* The resulting volume slices were compared against the corresponding lung masks in terms of the dice Coefficient value. Average dice value and minimuim dice values over all lung mask slices were used to evaluate the segemtnation method. <br/>

The codes are:
1. 'Region-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb'
2. 'Threshold-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb'
3. 'K_means_Clustering_based_Segmentation_using_COVID_19_CT_segmentation_dataset.ipynb"
<br/>
The table below shows a comparison between the three methods:

| **Segmentation Method**            |**Average Dice Coeffecient**|**Minimuim Dice Coeffecient**|
| -----------------------------------| ---------------------------|-----------------------------|
| `Region based`                     | 0.89311                    | 0.74504                     |
| ` histogram-Threshold (otsu) based`| 0.89370                    | 0.75117                     |
| `K-means clustering (K=2)`         | **0.89380**                | **0.75124**                 |
