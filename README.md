# DETECTO - The Object Detection And LandMark Identification App
Object Detection using YOLO and OpenCV and landmark identification using tensorflow. In this repository, I built a streamlit app for Object Detection, for both Video Object Detection and Image Object Detection landmark identification.
The Video , Images, Detected Videos, Detected Images and all Images used can be found in the Gallery directory.
This YOLO object Detection project can detect 80 objects(i.e classes) in either a video or image. The full list of the classes can be found in the labels directory as a textfile. Since a pretrained Model is made use of, the Yolo.v3 configuration can be found in the config_n_weights directory. YOLO.v3 weights is very heavy, about 200mb and should be downloaded. Click on this link to download the Yolo.v3 weights from here ![yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

![video](https://github.com/Divyansh6799/Objectdetection-web-application/blob/master/Gallary/video/traffic-jam-road-rage.mp4)
![Detected_video](https://github.com/Divyansh6799/Objectdetection-web-application/blob/master/Gallary/video/detected_traffic-jam-road-rage.mp4)
