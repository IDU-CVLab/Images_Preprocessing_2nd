This resporatory is attached to COV19D_2nd project. <br/>
The following lung segmentaion methods were used and their performance is measured using 'COVID-19 CT segmentation dataset':

# Region based segmentation using annotated public "COVID-19 CT segmentation dataset"
* The code 'Region-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb"
* The algorithm explores lung region based segmetnation.
* The annotated CT images at http://medicalsegmentation.com/covid19/ were used.
* From the annotated dataset: the Image volumes (308 Mb), and the Lung masks (1 Mb) were used to validate the segmentaiton performacne.
* The Dice Coeffecient was used to measure the preformance of the segmetnation method. Average dice value = 0.89311, minimuim value = 0.74504

# Histogram threshold segmentation using annotated public "COVID-19 CT segmentation dataset"
* The code 'Threshold-Based-Segmentation using COVID-19 CT segmentation dataset.ipynb"
* The algorithm explores lung otsu histogram based segmetnation.
* The annotated CT images at http://medicalsegmentation.com/covid19/ were used.
* From the annotated dataset: the Image volumes (308 Mb), and the Lung masks (1 Mb) were used to validate the segmentaiton performacne.
* The Dice Coeffecient was used to measure the preformance of the segmetnation method. Average dice value = 0.89370, minimuim value = 0.75117
