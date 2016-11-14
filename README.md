# Car Tracker
Given bounding box on a car in the first frame, track it in subsequent frames.

Two approaches were implemented:

1. To run histogram based tracking:
```
python src/tracker.py
```
This approach tends to fail as the orientation and scale of the car is constantly changing during the course of the video.
So, histogram constructed from the first frame is not robust enough.

2. To run detection based tracking:
```
python src/classify.py
```
We use cifar10 model trained with caffe for generating predictions. This approach tends to work reasonably well. 

Two future additions include:

1. Adding kalman filter
2. Weighing candidate templates with the distance from best detections in the last k frames

Time Estimates: Running this code on a macbook pro with 3GHz i7 processor and 16GB RAM takes ~2s per frame.
Note, if you have access to gpus and caffe compiled accordingly, setting caffe.set_mode_gpu() in Predictions._generate_net will give you considerable speedup.
