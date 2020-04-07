#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import cv2
import tkinter 
import PIL.Image, PIL.ImageTk

input_folder=r'D:\TEMP LAB\H2203'
file_name='2019-09-24 H2203 S1 free explorationDeepCut_resnet50_event_arena_bwDec5shuffle1_500000.csv'
vid_name='2019-09-24 H2203 S1 free explorationDeepCut_resnet50_event_arena_bwDec5shuffle1_500000_labeled.mp4'

output_folder=r'D:\TEMP LAB\DLC analysis\H2203'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
#1/FRAMERATE in sec
dT=0.040

#generates two vectors for X_Clean and Y_clean yielding the new positions in cm. if the likelihood is below T it will interpolate the x,y positions
#SB cohordinates is automatically assigned to the first low likelihood block
#SB x,y cohordinates
#range when the animal is in the Start Box (SB)

x_SB=380
y_SB=450
#in_the_startbox=25*int(input("When does the startbox open? (sec)"))
input("Did you insert the startbox intervals?")
input("Did you empty the extra list?")


#pixel size for BW videos (1 pixel = 0.311 cm)
pix=0.311

#treshold likelihood
L_treshold=0.4

DLC_data=pd.read_csv(os.path.join(input_folder, file_name), sep=',', index_col=0)
DLC_data=DLC_data[2:]
DLC_data.columns=['x', 'y', 'L']
DLC_data=DLC_data.astype(float)


#Insert here for intervals in the startbox in seconds
startbox=[]

TOTAL_SB=[]
for interval in startbox:
    TOTAL_SB=TOTAL_SB+list(range(interval[0]*25, interval[1]*25)) 

#initializes columns c and Speed
c=[0]
Speed=[0]

#remove frames when the animal is in the startbox
for i in TOTAL_SB:
    DLC_data.iloc[i]=DLC_data.iloc[i]*0+[x_SB, y_SB, 1]

#DLC_data=DLC_data.iloc[range(in_the_startbox),in_the_startbox_2,in_the_startbox_3,in_the_startbox_4,in_the_startbox_5]*0 + [x_SB, y_SB, 1]

#df1=DLC_data.iloc[range(in_the_startbox)+in_the_startbox_2+in_the_startbox_3+in_the_startbox_4+in_the_startbox_5]*0 + [x_SB, y_SB, 1]
#df2=DLC_data[in_the_startbox:25*63]+DLC_data[25*94:25*102]+DLC_data[25*141:25*158]+DLC_data[25*246:328*25]+DLC_data[25*383:]
#DLC_data = pd.concat([df1, df2])

#IMPORTANT: checked that the number this maintains the number of rows

X=DLC_data['x']
Y=DLC_data['y']
for i in DLC_data.index[1:]:
    i= int(i)
    Speed.append((math.sqrt((X[i] - X[i-1])*(X[i] - X[i-1])+(Y[i] - Y[i-1])*(Y[i] - Y[i-1])))/dT)
    c.append(i/len(DLC_data.index))
Speed=np.asarray(Speed)

#appends two columns for color code and Speed
DLC_data.insert(2, 'Speed', Speed*pix)
DLC_data.insert(0, 'c', c)

DLC_data=DLC_data.mask(DLC_data['L'] < L_treshold, 0)

DLC_data


# In[ ]:


#flags points where the animal is moving at > 1.0 m/s and inserts them in a list (frames_to_check)
#max speed allowed

speed_max = 100
frames_to_check =[]

for i in DLC_data.index:
    if DLC_data['Speed'][i] > speed_max:
        i=int(i)
        frames_to_check.append(i)

#if two frames in frames_to_check are closer than 5 frames, it will also include inbetween frames
for j in range(len(frames_to_check)):
    if int(frames_to_check[j] - frames_to_check[j-1]) < 5:
        frames_to_check = frames_to_check + [k for k in range(int(frames_to_check[j-1]),int(frames_to_check[j]+1))]

#removes duplicates
frames_to_check = list(dict.fromkeys(frames_to_check))

wrong_frames=[]

#outputs the number of frames to check
print("You have", len(frames_to_check), "frames to check.")


# In[ ]:


# WARNING! ONLY RUN IF CRASHED. THIS CAN LOAD PREVIOUS SAVED FRAMES

#--------------------------------------------------

with open('E:\DLC\DLC analysis\H2203\wrong_temp2.txt') as f:
    wrong_frames = f.read().splitlines()

for i in range(0, len(wrong_frames)): 
    wrong_frames[i] = int(wrong_frames[i]) 

#--------------------------------------------------

extra=[]

extra1=[]

for interval in extra1:
    extra=extra+list(range(interval[0], interval[1])) 
    
wrong_frames=wrong_frames +extra 

#--------------------------------------------------

frames_to_check=[i for i in frames_to_check if i not in wrong_frames]

print("You have", len(frames_to_check), "frames to check.")


# In[ ]:


#divides the frmaes to check in two block and allows to manually select them

#opens the video
cap = cv2.VideoCapture(os.path.join(input_folder, vid_name))

if (cap.isOpened()== False): 
    print("Error opening video file")

##creates a txt for temporary storage of frames that are labeled as incorrect
wrong_temp_file= open(os.path.join(output_folder, "wrong_temp" + ".txt"),"w+")

#creates a list of frames that are labeled as incorrect
wrong=[]

#visualizes the corresponding frame of the video with suspect speed. Frames can be accepted (y) or dropped (n)
for frame in frames_to_check[:100]:
 
    #initializes window in Tkinter and displays the frames to check
    window = tkinter.Tk()
    cap.set(1, int(frame))
    ret, img = cap.read()
    gray = img
    height, width, channel = gray.shape
    canvas = tkinter.Canvas(window, width = width, height = height)
    canvas.pack()
    frame_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(gray))
    canvas.create_image(0, 0, image=frame_photo, anchor=tkinter.NW)
    
    print("\n \n Displayed frame : ", frame, "               ", frames_to_check.index(frame), "/", len(frames_to_check))
    
    #opens Tkinter window
    window.update()
    
    #asks for input whether the displayed frame is correct or not
    correct_label = input("Is this frame labelled correctly - y/n? Enter 'q' to Quit  : ")

    if correct_label== 'n':
        wrong.append(int(frame))
        window.destroy()
        wrong_temp_file= open(os.path.join(output_folder, "wrong_temp" + ".txt"),"a+")
        wrong_temp_file.write('%i \n' % frame)
        wrong_temp_file.close()
        
    elif correct_label=='q':
        print("Process interrupted")
        window.destroy()
        break
        
    else:
        window.destroy()
        
wrong_temp_file.close()        


# In[ ]:





# In[ ]:


#updates the wrong frames into a stable list and empties the temporary list
wrong_frames= wrong
wrong=[]

for frame in frames_to_check[100:]:
    #initializes window in Tkinter and displays the frames to check
    window = tkinter.Tk()
    cap.set(1, int(frame))
    ret, img = cap.read()
    gray = img 
    height, width, channel = gray.shape
    canvas = tkinter.Canvas(window, width = width, height = height)
    canvas.pack()
    frame_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(gray))
    canvas.create_image(0, 0, image=frame_photo, anchor=tkinter.NW)
    
    print("\n \n Displayed Frame : ", frame, "               ", frames_to_check.index(frame), "/", len(frames_to_check))
    
    #opens Tkinter window
    window.update()
    
    #asks for input whether the displayed frame is correct or not
    correct_label = input("Is this frame labelled correctly - y/n? Enter 'q' to Quit  : ")

    if correct_label== 'n':
        wrong.append(int(frame))
        window.destroy()
        wrong_temp_file= open(os.path.join(output_folder, "wrong_temp" + ".txt"),"a+")
        wrong_temp_file.write('%i \n' % frame)
        wrong_temp_file.close()
        
    elif correct_label=='q':
        print("Process interrupted")
        window.destroy()
        break
        
    else:
        window.destroy()
        
      


# In[ ]:





# In[ ]:


#--------------------------------------------------
with open('D:\TEMP LAB\DLC analysis\H2203\wrong_temp.txt') as f:
    wrong_frames = f.read().splitlines()
wrong_frames

for i in range(0, len(wrong_frames)): 
    wrong_frames[i] = int(wrong_frames[i]) 
wrong_frames
#--------------------------------------------------


# In[ ]:


wrong=[]
wrong_frames= wrong_frames + wrong
#final vectors for x,y are generated. If a frame is in the wrong_frames list, it will be dropped and the x,y coordinates interpolated
X_sort=[]
Y_sort=[]
i=0

WRONG=np.asarray(wrong_frames)

for i in range(len(DLC_data.index)):
    if int(i) in WRONG:
        X_sort.append(0)
        Y_sort.append(0)
        i=+1
    else: 
        X_sort.append(DLC_data['x'][i])
        Y_sort.append(DLC_data['y'][i])
        i=+1

DLC_not_too_fast=pd.DataFrame(data={'x': X_sort, 'y': Y_sort,  'L':DLC_data['L']})



# In[ ]:


#recalculates the speed for the updated coordinates

Speed_not_too_fast=[0]
X3=DLC_not_too_fast['x']
Y3=DLC_not_too_fast['y']

for i in DLC_not_too_fast.index[1:]:
    i= int(i)
    Speed_not_too_fast.append((math.sqrt((X3[i] - X3[i-1])*(X3[i] - X3[i-1])+(Y3[i] - Y3[i-1])*(Y3[i] - Y3[i-1])))/dT)
    
Speed_not_too_fast=np.asarray(Speed_not_too_fast)
DLC_not_too_fast.insert(2, 'Speed', Speed_not_too_fast*pix)
DLC_not_too_fast.insert(0, 'c', c)

#resorting to see if there's more suspicious frames
frames_to_check_again=[]

for i in DLC_not_too_fast.index:
    if DLC_not_too_fast['Speed'][i] > speed_max:
        i=int(i)
        frames_to_check_again.append(i)

#if two frames in frames_to_check are closer than 25 frames, it will also include inbetween frames
for j in range(len(frames_to_check_again)):
    if int(frames_to_check_again[j] - frames_to_check_again[j-1]) < 15:
        frames_to_check_again = frames_to_check_again + [k for k in range(int(frames_to_check_again[j-1]),int(frames_to_check_again[j]))]

frames_to_check_again = list(dict.fromkeys(frames_to_check_again))        
frames_to_check_again = [item for item in frames_to_check_again if item not in frames_to_check ]  
frames_to_check_again = [item for item in frames_to_check_again if item not in wrong_frames ]  


# In[ ]:


#import cv2

wrong_again=[]

cap = cv2.VideoCapture(os.path.join(input_folder, vid_name))

#visualizes the corresponding frame of the video with suspect speed. Frames can be accepted (y) or dropped (n)
if len(frames_to_check_again) == 0:
    print("Horray! No more frames to check!")
else:
    
    print("You have", len(frames_to_check_again), "more frames to check")
    
    for frame in frames_to_check_again:
       
        #initializes window in Tkinter and displays the frames to check
        window = tkinter.Tk()
        cap.set(1, int(frame))
        ret, img = cap.read()
        gray = img 
        height, width, channel = gray.shape
        canvas = tkinter.Canvas(window, width = width, height = height)
        canvas.pack()
        frame_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(gray))
        canvas.create_image(0, 0, image=frame_photo, anchor=tkinter.NW)

        print("\n \n Displayed Frame : ", frame, "               ", frames_to_check_again.index(frame), "/", len(frames_to_check_again))

        #opens Tkinter window
        window.update()

        #asks for input whether the displayed frame is correct or not
        correct_label = input("Is this frame labelled correctly - y/n? Enter 'q' to Quit  : ")

        if correct_label== 'n':
            wrong.append(int(frame))
            window.destroy()
            wrong_temp_file= open(os.path.join(output_folder, "wrong_temp" + ".txt"),"a+")
            wrong_temp_file.write('%i \n' % frame)
            wrong_temp_file.close()

        elif correct_label=='q':
            print("Process interrupted")
            window.destroy()
            break

        else:
            window.destroy()
        


# In[ ]:





# In[ ]:


#optional: additional frames can be added here (for example if by mistake the wrong answer was given. 
#Can also be run post-selection, it is back-compatible)

wrong_frames= wrong_frames + wrong_again

#insert extra frames
extra=[]

#insert ranges in the format [(begin, end), (begin, end), ...]
extra_list=[]
if extra_list != []:
    for interval in extra_list:
        extra=extra+list(range(interval[0], interval[1])) 
else:
    extra=extra+extra_int

wrong_frames_2=wrong_frames + extra 

#removes duplicates
wrong_frames_2 = list(dict.fromkeys(wrong_frames_2))

wrong_frames_2.sort()


# In[ ]:


X2=[]
Y2=[]
i=0
wrong_frame=np.asarray(wrong_frames_2)
for i in range(len(DLC_data.index)):
    if int(i) in wrong_frame:
        X2.append(0)
        Y2.append(0)
        i=+1
    else: 
        X2.append(DLC_data['x'][i])
        Y2.append(DLC_data['y'][i])
        i=+1

DLC_not_too_fast2=pd.DataFrame(data={'x': X2, 'y': Y2,  'L':DLC_data['L']})

DLC_not_too_fast2[DLC_not_too_fast2==0] = np.nan


# In[ ]:


DLC_not_too_fast2=DLC_not_too_fast2.interpolate(method='linear', limit_direction='forward', axis=0)

Speed_not_too_fast=[0]
X3=DLC_not_too_fast2['x']
Y3=DLC_not_too_fast2['y']

for i in DLC_not_too_fast.index[1:]:
    i= int(i)
    Speed_not_too_fast.append((math.sqrt((X3[i] - X3[i-1])*(X3[i] - X3[i-1])+(Y3[i] - Y3[i-1])*(Y3[i] - Y3[i-1])))/dT)
    
Speed_not_too_fast=np.asarray(Speed_not_too_fast)
DLC_not_too_fast2.insert(2, 'Speed', Speed_not_too_fast*pix)
DLC_not_too_fast2.insert(0, 'c', c)


# In[ ]:


#allows you to check selected intervals before and after the cleanup process
DLC_not_too_fast2[DLC_not_too_fast2['c']>0.8].plot.scatter('x', 'y', c ='red', s=2)

DLC_data.plot.scatter('x', 'y', c ='c', s=2)

DLC_not_too_fast2.plot.scatter('x', 'y', c ='c', s=2)

#DLC_not_too_fast[3840:3900]


# In[ ]:


#allows inspetion of the cleaned speed to see if there's any suspect frame to check

DLC_data['Speed'].plot(kind='line', c= 'lightgrey')

DLC_not_too_fast2['Speed'].plot(kind='line', c= 'red')
plt.xlim(0,20000)
plt.ylim(0,600)


# In[ ]:


cap = cv2.VideoCapture(os.path.join(input_folder, vid_name))

frame = int(14662)
cap.set(1, int(frame))
ret, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray_blurred = cv2.blur(gray, (3, 3)) 
# Saves the frames with frame-count 
plt.imshow(img)  
plt.show()

print("Frame : ", frame)


# In[ ]:





# In[ ]:


#calculates the smoothed speed using a sliding window. 
#Position 'i' is smoothed by averaging 'i' with the previous 4 frames and 'i-1' is the average of the previous 5 frames. 
#Smoothed spees is then calculated. dT is the same. 

Speed_smooth=[0,0,0,0,0]

for i in DLC_not_too_fast2.index[5:-5]:
    i= int(i)
    X_smooth=np.mean(DLC_not_too_fast2['x'][i:i+4])
    Y_smooth=np.mean(DLC_not_too_fast2['y'][i:i+4])
    X_smooth_previous=np.mean(DLC_not_too_fast2['x'][i-5:i-1])
    Y_smooth_previous=np.mean(DLC_not_too_fast2['y'][i-5:i-1])
      
    Speed_smooth.append((math.sqrt((X_smooth - X_smooth_previous)*(X_smooth - X_smooth_previous)+(Y_smooth - Y_smooth_previous)*(Y_smooth - Y_smooth_previous)))/(5*dT))

Speed_smooth.append(0)
Speed_smooth.append(0)
Speed_smooth.append(0)
Speed_smooth.append(0)
Speed_smooth.append(0)

Speed_smooth=np.asarray(Speed_smooth)
DLC_not_too_fast2.insert(3, 'Speed_smooth', Speed_smooth*pix)
DLC_not_too_fast2.insert(3, 'x_cm', DLC_not_too_fast2['x']*pix)
DLC_not_too_fast2.insert(4, 'y_cm', DLC_not_too_fast2['y']*pix)


# In[ ]:


DLC_not_too_fast2[8740:8760]


# In[ ]:


DLC_not_too_fast2['Speed'].plot(kind='line', c= 'lightgrey')

DLC_not_too_fast2['Speed_smooth'].plot(kind='line', c= 'red')
plt.xlim(1000,2000)
plt.ylim(0,100)


# In[ ]:


new_name='H2203_N01_SW2_PRE'
    
#saves the output
WF=pd.DataFrame(data={'WRONG FRAMES' : wrong_frames_2})

DLC_EXPORT_FILE= os.path.join(output_folder, new_name + '.csv')
FRAMES_EXPORT_FILE = os.path.join(output_folder, new_name + "_frames" + '.csv')

if os.path.exists(DLC_EXPORT_FILE):
    print('WARNING: File already exists (Moron)')

else:
    DLC_not_too_fast2.to_csv(DLC_EXPORT_FILE, sep=',')
    WF.to_csv(FRAMES_EXPORT_FILE, sep=',')
    
if os.path.exists(DLC_EXPORT_FILE) and os.path.exists(FRAMES_EXPORT_FILE) :
    print('Files saved successfully')


# In[ ]:


cap.release()
#video's should all have same dim - if not true:
#width=cap.get(3)
#height=cap.get(4)
    
width = 720 
height = 576
FPS = 25

cap = cv2.VideoCapture(os.path.join(input_folder, vid_name))

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
video=cv2.VideoWriter(os.path.join(output_folder,  new_name  + '_overlay.mp4'), fourcc, int(FPS), (int(width), int(height)))

end_vid= len(DLC_not_too_fast2.index)

i=0


for i in range(end_vid):
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    text = 'Frame number: ' + str(cap.get(1))
    new=cv2.putText(frame, text, (20,540) , font, 1, (0,255,0), 2)
    if i > 100:
        for k in range(i-99,i):
            new = cv2.circle(new, (int(DLC_not_too_fast2['x'][k]), int(DLC_not_too_fast2['y'][k])), 2, (255, 255, 0), -1)
    newer = cv2.circle(new, (int(DLC_not_too_fast2['x'][i]), int(DLC_not_too_fast2['y'][i])), 6, (0, 225, 0), -1)
    video.write(newer)

    
video.release()
cap.release()
    


# In[ ]:





# In[ ]:




