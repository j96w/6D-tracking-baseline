# 6D-tracking-baseline

(Under Construction)

This is a basic model-free rigid object 6D pose tracking baseline with ICP(w/wo color registration) and truncated signed distance function(TSDF) fusion.
* The ICP w/wo color registration is from [Open3D](http://www.open3d.org/).
* The TSDF fusion is adapted from [andyzeng/tsdf-fusion-python](https://github.com/andyzeng/tsdf-fusion-python).
* Also, the idea of this code is partly inspired by [Co-Fusion](http://visual.cs.ucl.ac.uk/pubs/cofusion/).

Thanks for these great projects.

Basically, this code is trying to conduct 6D pose tracking and object model reconstruction at the same time. After given the GT pose or 3D bbox of the target object in the first frame, the code will start to track the 6D pose changes of the object in each video frame with ICP(w/wo color registration) and do TSDF fusion to reconstruct the mesh after a certain number of steps of tracking. No object mesh model is required to run this code. Several experiments have been done in these datasets
* [YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/)
* YCB_synthetic_video_dataset. (Can be generated with [j96w/sixd_toolkit](https://github.com/j96w/sixd_toolkit) adapted from [thodan/sixd_toolkit](https://github.com/thodan/sixd_toolkit))
* [redwood-dataset](http://redwood-data.org/3dscan/)

Some result demo:

YCB_synthetic_video_dataset

![image](https://github.com/j96w/6D-tracking-baseline/blob/master/demo/ycb_syn.gif)