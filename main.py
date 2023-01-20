import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys

def object_detection_video():
    #object_detection_video.has_beenCalled = True
    #pass
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    font_scale = 1
    thickness = 1
    url = "https://raw.githubusercontent.com/Divyansh6799/Objectdetection-web-application/50aeb4cdf52b1ef00442ca9086b1ceebf71357d6/labels/coconames.txt"
    f = urllib.request.urlopen(url)
    labels = [line.decode('utf-8').strip() for  line in f]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    st.title("Object Detection for Videos")
    st.subheader("""
    This object detection App takes in a video and outputs the video with bounding boxes created around the objects in the video 
    """
    )
    st.write("The Optimal Video Length is 30sec or less For Better and accurate Detection.")
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        text_1 = st.markdown("It takes few minutes to give output, Please Wait.....")
        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        print(image)
        h, w = image.shape[:2]
        #out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc#(*'avc3'), fps, insize)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
        count = 0
        while True:
            _, image = cap.read()
            if _ != False:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count +=1
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # loop over each of the layer outputs
                for output in layer_outputs:
                    # loop over each of the object detections
                    for detection in output:
                        # extract the class id (label) and confidence (as a probability) of
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # discard weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # perform the non maximum suppression given the scores defined before
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
                font_scale = 0.6
                thickness = 1
                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = image.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                        cv2.putText(image, "press q to exit", (10,20), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0,0,255), thickness=4)

                out.write(image)
                cv2.imshow("Detection", image)
                if ord("q") == cv2.waitKey(1):
                    break
            else:
                break

        text_1.empty()
        cap.release()
        cv2.destroyAllWindows()
          
#function for object detection
def obj_detection_image():
    st.title('Object Detection For Images')
    st.subheader("This object detection App takes the input as image and outputs the image with objects bounded in a rectangle with confidence score.")
    uploaded_file = st.file_uploader("Upload a image",type='jpg')
    if uploaded_file != None:
        image1 = Image.open(uploaded_file)
        image2 =np.array(image1)
        st.image(image1, caption='Uploaded Image.')
        text = st.markdown("It takes few minutes to give output, Please Wait.....")
        my_bar = st.progress(0)
        confThreshold =st.slider('Confidence', 0, 100, 50)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        whT = 320
        #### LOAD MODEL
        ## Coco Names
        classesFile = "coco.names"
        classNames = []
        with open(classesFile, 'rt') as f:
            classNames = f.read().split('\n')
            
        ## Model Files        
        modelConfiguration = "yolov3.cfg"
        modelWeights = "https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights"
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        #finding the objects
        def findObjects(outputs,img):
            hT, wT, cT = image2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
            obj_list=[]
            confi_list =[]
            #drawing rectangle around object
            for i in indices:
                i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(image2, (x, y), (x+w,y+h), (255, 0 , 255), 2)
                #print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(image2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
            if st.checkbox("Show Object's list" ):
                st.write(df)
            if st.checkbox("Show Confidence bar chart" ):
                st.subheader('Bar chart for confidence levels')
                st.bar_chart(df["Confidence"])
           
        blob = cv2.dnn.blobFromImage(image2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,image2)
        text.empty()
        st.image(image2, caption='Proccesed Image.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        my_bar.progress(100)
        
#function for landmark identification
def landmark_detection() :
    st.title('Landmark identification')
    st.subheader("This takes the input image and identifies the landmark in the image from perticular Region.[so Select Region from below].")
    chs_region  = st.selectbox("Choose Region",("Asia","Africa","Europe","North America","South America","Oceania & Antarctica"))
    uploaded_file = st.file_uploader("Upload a image",type=['jpg','jfif','png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.')
        text_2 = st.markdown("It takes few minutes to give output, Please Wait.....")
        my_bar = st.progress(0)
        TF_MODEL_URL=""
        LABEL_MAP_URL=""
        if chs_region=="Asia":
            TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
            LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
        elif chs_region=="Africa":
            TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_africa_V1/1'
            LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_africa_V1_label_map.csv'
        elif chs_region=="Europe":
            TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_europe_V1/1'
            LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_europe_V1_label_map.csv'
        elif chs_region=="North America":
            TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_north_america_V1/1'
            LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_north_america_V1_label_map.csv'
        elif chs_region=="South America":
            TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_south_america_V1/1'
            LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_south_america_V1_label_map.csv'
        elif chs_region=="Oceania & Antarctica":
            TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_oceania_antarctica_V1/1'
            LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_oceania_antarctica_V1_label_map.csv'
        IMAGE_SHAPE = (321, 321)
        classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                 input_shape=IMAGE_SHAPE+(3,),
                                                 output_key="predictions:logits")])
        df = pd.read_csv(LABEL_MAP_URL)
        label_map = dict(zip(df.id, df.name))
        img = image.resize(IMAGE_SHAPE)
        img = np.array(img)/255.0
        img = img[np.newaxis, ...]
        prediction = classifier.predict(img)
        text_2.empty()
        st.header(label_map[np.argmax(prediction)])
        my_bar.progress(100)

def main():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://plus.unsplash.com/premium_photo-1670659359754-02934f07580f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTN8fHRlY2hub2xvZ3klMjBiYWNrZ3JvdW5kfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60");
    background-size: 100%;
    background-position: top ;
    background-repeat: repeat;
    background-attachment: local;
    }}
    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("https://images.unsplash.com/photo-1470811976196-8ee4fa278c5d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxjb2xsZWN0aW9uLXBhZ2V8Mnw0NzI2NTI5fHxlbnwwfHx8fA%3D%3D&auto=format&fit=crop&w=500&q=60");
    background-size: 100%;
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    new_title = '<p style="font-size: 80px; color:blue">DETECTO</p>''<p style="font-size: 30px;">Welcome to The Object Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    This App was built using Streamlit and OpenCV 
    to demonstrate YOLO Object detection in both videos(pre-recorded)
    & images, Also Identify the Landmarks Of the world.
    This YOLO object Detection project can detect 80 objects(i.e classes)
    in either a video or image. The full list of the classes can be found 
    [here](https://github.com/Divyansh6799/Objectdetection-web-application/blob/master/labels/coconames.txt).

    Select Option To Try Features in Sidebar which present On Left .....

    Developed By [Divyansh Trivedi](https://divyanshtrivediportfolio.netlify.app/) 

    Follow Us On:
        [Github](https://github.com/Divyansh6799/) &
        [Linkedin](https://www.linkedin.com/in/divyansh-trivedi-1551581bb/)
        """
    )
    st.sidebar.title("DETECTO")
    choice  = st.sidebar.selectbox("Select OPTION",("About","Object Detection(Image)","Object Detection(Video)","Landmark identification"))
    read=st.sidebar.markdown("""
    This App was built using Streamlit and OpenCV 
    to demonstrate YOLO Object detection in both videos(pre-recorded)
    and images.This YOLO object Detection project can detect 80 objects(i.e classes)
    in either a video & images, Also Identify the Landmarks Of the world. The full list of the classes can be found 
    [here](https://github.com/Divyansh6799/Objectdetection-web-application/blob/master/labels/coconames.txt).

    Developed By [Divyansh Trivedi](https://divyanshtrivediportfolio.netlify.app/) 

    Follow Us On:
        [Github](https://github.com/Divyansh6799/) &
        [Linkedin](https://www.linkedin.com/in/divyansh-trivedi-1551581bb/)
        """
    )
    if choice == "Object Detection(Image)":
        #st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('Object Detection')
        obj_detection_image()
    elif choice == "Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()
        #object_detection_video.has_beenCalled = False
        object_detection_video()
        #if object_detection_video.has_beenCalled:
        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video") 
        except OSError:
            ''

    elif choice == "Landmark identification":
        read_me_0.empty()
        read_me.empty()
        landmark_detection() 

    elif choice == "About":
        print()
        read.empty()
        

if __name__ == '__main__':
		main()	