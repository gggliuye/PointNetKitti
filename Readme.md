# Files

## MakeDataBaseFromKittiObjectDection.ipynb

Make data set from the ground truth.

* Decode the KITTI ground truth result.
* Cut the segment of cloud that contains each object -> to make an object classification dataset.
<div align="center">    
<img src="images/bbx.PNG" width="70%" height="70%" />
</div>
* Use mine implemenataion of clustering algorihm to test the lidar cloud segmentation.
<div align="center">    
<img src="images/clustering_result_3.PNG" width="80%" height="80%" />
</div>

## PretreatmentDataSetAndClassification.ipynb

* Pretreatment of the raw data, and build a PointNet++ to classify them. (Classfication precision reaches 90%)
Mine model could be found [here in Baidu Yun](https://pan.baidu.com/s/1GrUqz7I0CsjB_nDMYcTF2w) with extraction code '5u6q' .

<div align="center">    
<img src="images/acc.PNG" width="80%" height="80%" />
</div>

* Implement the model to the whole KITTI set for detection task. (Traditional segmentation + Pointnet++ classification)

<div align="center">    
<img src="images/test_result.PNG" width="80%" height="80%" />
</div>

## ResultEvaluation.ipynb

* Analysis the results, show the recall and precision.
* What todo in the future.
