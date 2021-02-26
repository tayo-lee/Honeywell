# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 03:36:30 2021

@author: jaiha
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from datetime import datetime
from scipy.spatial import distance
import numpy as np
import pandas as pd
import imutils
import time
import cv2
import os
from tkinter import *
import pkg_resources.py2_warn
from centroidtracker import CentroidTracker

tracker = CentroidTracker(maxDisappeared=5, maxDistance=90)

def option():
    print(clicked.get())
    if clicked.get()=="Video":
        mainvideosave()
    elif clicked.get()=="Photo":
        mainimagesave()
    elif clicked.get()=="Webcam":
        mainvideosavecam()

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))    
    faceNet.setInput(blob)
    detections = faceNet.forward()    
    faces = []
    locs = []
    preds2 = []
    new=[]
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            #face = frame[startY:endY, startX:endX]
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            #face = cv2.resize(face, (224, 224))
            #face = img_to_array(face)
            #face = preprocess_input(face)
            #face = np.expand_dims(face, axis=0)
            #faces.append(face)
            locs.append((startX, startY, endX, endY))
    #if len(faces) > 0:
        #for i in range(len(faces)):
            #preds2.append(maskNet.predict(faces))
    #new = zip(locs, preds2)
    return locs

def mainvideosave():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(os.path.sep.join(["face_detector", "mask_detector2.h5"]))
    vs = FileVideoStream("video.mp4").start()
    time.sleep(2.0)
    i=0
    dict={}
    while(i==0):
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        height = (np.shape(frame))[0]
        i+=1
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (400,height))
    while True:
        rects = []
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        new = detect_and_predict_mask(frame, faceNet, maskNet)
        now = datetime.now()
        date_string = now.strftime("%d/%m/%Y")
        time_string = now.strftime("%H:%M:%S")
        for box in new:
            (startX, startY, endX, endY) = box
            #mask = pred[0][0]
            #withoutmask = pred[0][1]        
            #label = "Mask" if mask > withoutmask else "No mask"
            #color = (0, 255, 0) if label == "Mask" else (0, 0, 255)        
            #label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
            #cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,255), 2)
            
            rects.append(box)
        
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        predictions=[]
        faces=[]
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            #print(len(objects.items()))
            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            #faces.append(face)
            pred = maskNet.predict(face)
            #print(predictions)
            mask = pred[0][0]
            withoutmask = pred[0][1]        
            label = "Mask" if mask > withoutmask else "No mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)        
            label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            if i==1:
                dict.update({'Number':[i], 'Date':date_string, 'Time':time_string, 'Label':label})
                df = pd.DataFrame(dict)
                df.to_csv(r'recordfile_vid.csv', index=False)
            elif i>=1:
                dict.update({'Number':[i], 'Date':date_string, 'Time':time_string, 'Label':label})
                df = pd.DataFrame(dict)
                df.to_csv(r'recordfile_vid.csv',mode='a', header=False, index=False)
            i+=1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            out.release()
            cv2.destroyAllWindows()
            vs.stop()
            break
    out.release()
    cv2.destroyAllWindows()
    vs.stop()

def mainvideosavecam():
    j=1
    dict={}
    previds=[-1]
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'frames_cam')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(os.path.sep.join(["face_detector", "mask_detector2.h5"]))
    vs = VideoStream(0).start()
    time.sleep(2.0)
    while True:
        rects = []
        frame = vs.read()
        #frame = imutils.resize(frame, width=1080)
        new = detect_and_predict_mask(frame, faceNet, maskNet)
        now = datetime.now()
        date_string = now.strftime("%d/%m/%Y")
        time_string = now.strftime("%H:%M:%S")
        for box in new:
            (startX, startY, endX, endY) = box
            #mask = pred[0][0]
            #withoutmask = pred[0][1]        
            #label = "Mask" if mask > withoutmask else "No mask"
            #color = (0, 255, 0) if label == "Mask" else (0, 0, 255)        
            #label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
            #cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,255), 2)
            
            
            rects.append(box)
            
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        predictions=[]
        faces=[]
        ids=[]
        objects = tracker.update(rects)
        
        for (objectId, bbox) in objects.items():
            ids.append(objectId)
            
        for (objectId, bbox) in objects.items():
            
            try:
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                #print(previds)
                #print(len(objects.items()))
                face = frame[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                #faces.append(face)
                pred = maskNet.predict(face)
                #print(predictions)
                mask = pred[0][0]
                withoutmask = pred[0][1]        
                label = "Mask" if mask > withoutmask else "No mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)        
                label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = "ID: {}".format(objectId)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                
                for id1 in ids:
                    k=0
                    l=0
                    for id2 in previds:
                        if id1 == id2:    
                            break
                        elif id1 in previds:
                            k=1
                        elif id1>id2 and k==0:
                            #print(id1)
                            cv2.imwrite(os.path.join(final_directory,str(id1)+".jpg"), frame[startY:endY, startX:endX])
                            if j==1:
                                dict.update({'Number':[j], 'Date':date_string, 'Time':time_string, 'Label':label})
                                df = pd.DataFrame(dict)
                                df.to_csv(r'recordfile_videocam.csv', index=False)
                            else:
                                dict.update({'Number':[j], 'Date':date_string, 'Time':time_string, 'Label':label})
                                df = pd.DataFrame(dict)
                                df.to_csv(r'recordfile_videocam.csv',mode='a', header=False,index=False)
                            j+=1                        
                            previds.append(id1)
                        elif id2 in ids:
                            #print("yes")
                            l=1
                        elif l==0:
                            #print("yes")
                            previds.remove(id2)
                            
                    for id1 in ids:
                        l=0
                        for id2 in previds:
                            if id2 in ids:
                                #print("yes")
                                l=1
                            elif l==0:
                                #print("yes")
                                previds.remove(id2)
                                
            except:
                continue                        
        cv2.imshow("Frame", frame)     
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break
    cv2.destroyAllWindows()
    vs.stop()

def mainimagesave():
    j=1
    dict={}
    previds=[-1]
    No_of_violations=0
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'frames')
    count_mask=0 # this line is added by me
    count_no_mask=0
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(os.path.sep.join(["face_detector", "mask_detector2.h5"]))
    
    #sd = social distancing
    protox_sd = "frozen_inference_graph.pb"
    model_sd = "graph.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(protox_sd, model_sd)
    
    (H, W) = (None, None)
    #sd
    
    vs = FileVideoStream("video.mp4").start()
    time.sleep(2.0)
    while True:
        rects = []
        rects_sd = [] 
        centroids=[]
        distdata=[]
        counter=0
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        """
        1
        """
        #bounding box for sd
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects_sd.append(box.astype("int"))
                (startX_sd, startY_sd, endX_sd, endY_sd) = box.astype("int")
                centroids.append(startX_sd+(int((endX_sd-startX_sd)/2), startY_sd+(int((endY_sd-startY_sd)/2))))
                cv2.rectangle(frame, (startX_sd, startY_sd), (endX_sd, endY_sd), (0,255,0), 2)
        
        for i in range(len(centroids)-1):
            for j in range(i+1,len(centroids)):
                distdata.append([i,j,distance.euclidean(centroids[i], centroids[j])])    
        for (i,j,distancee) in distdata:
            if(distancee<50):
                cv2.rectangle(frame,(rects_sd[i][0],rects_sd[i][1]),(rects_sd[i][2], rects_sd[i][3]),(0,0,255),2)
                cv2.rectangle(frame,(rects_sd[j][0],rects_sd[j][1]),(rects_sd[j][2], rects_sd[j][3]),(0,0,255),2)
                No_of_violations+=1
                
            else:
                continue
        
        """
        1
        """        
        new = detect_and_predict_mask(frame, faceNet, maskNet)
        now = datetime.now()
        date_string = now.strftime("%d/%m/%Y")
        time_string = now.strftime("%H:%M:%S")
        
        if counter%20 == 0:
            No_of_violations = 0
            count_mask=0
            count_no_mask=0
            
        for box in new:
            (startX, startY, endX, endY) = box
            rects.append(box)
            
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        predictions=[]
        faces=[]
        ids=[]
        objects = tracker.update(rects)
        
        for (objectId, bbox) in objects.items():
            ids.append(objectId)
            
        for (objectId, bbox) in objects.items():
            
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
                       
            #print(previds)
            #print(len(objects.items()))
            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            #faces.append(face)
            pred = maskNet.predict(face)
            #print(predictions)
            mask = pred[0][0]
            withoutmask = pred[0][1]        
            label = "Mask" if mask > withoutmask else "No mask"
            """
            if mask> withoutmask: # this line is added by me
                count_mask+=1 # this line is added by me
            else: # this line is added by me
                count_no_mask+=1 # this line is added by me
            """
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)        
            label1 = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
            cv2.putText(frame, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            Id_and_label=[objectId, label]
            
            #print(ids)
            for id1 in ids:
                k=0
                l=0
                for id2 in previds:
                    if id1 == id2:    
                        break
                    elif id1 in previds:
                        k=1
                    elif id1>id2 and k==0:
                        #print(id1)
                        if Id_and_label[1]=="Mask":
                            count_mask+=1
                        elif Id_and_label[1]=="No mask":
                            count_no_mask+=1
                        cv2.imwrite(os.path.join(final_directory,str(id1)+".jpg"), frame[startY:endY, startX:endX])
                        if j==1:
                            dict.update({'Number':[j], 'Date':date_string, 'Time':time_string, 'Label':label1, 'No_Mask_count':count_mask,'Mask_Count':count_no_mask, "Violation_count":No_of_violations})
                            df = pd.DataFrame(dict)
                            df.to_csv(r'recordfile_img.csv', index=False)
                        else:
                            dict.update({'Number':[j], 'Date':date_string, 'Time':time_string, 'Label':label1, 'No_Mask_count':count_mask,'Mask_Count':count_no_mask, "Violation_count":No_of_violations})
                            df = pd.DataFrame(dict)
                            df.to_csv(r'recordfile_img.csv',mode='a', header=False,index=False)
                        j+=1                        
                        previds.append(id1)
                    elif id2 in ids:
                        #print("yes")
                        l=1
                    elif l==0:
                        #print("yes")
                        previds.remove(id2)
            
            for id1 in ids:
                l=0
                for id2 in previds:
                    if id2 in ids:
                        #print("yes")
                        l=1
                    elif l==0:
                        #print("yes")
                        previds.remove(id2)
            counter+=1
            #print(previds)
        cv2.imshow("Frame", frame)     
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break
    cv2.destroyAllWindows()
    vs.stop()
    
window = Tk()
window.title("My computer")
window.configure(background="black")
photo1 = PhotoImage(file="source.gif")
Label (window, image=photo1, bg="black") .grid(row=0, column=0, sticky=E)
clicked = StringVar()
clicked.set("Video")
drop = OptionMenu(window, clicked, "Video", "Photo", "Webcam")
drop.grid(row=1, column=0, sticky=W)
mainbutton = Button(window, text="Detection", command=option)
mainbutton.grid(row=2, column=0, sticky=W)
window.mainloop()