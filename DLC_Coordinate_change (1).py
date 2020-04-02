#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import cv2
import numpy as np
from numpy import *
import scipy.signal 
from scipy import stats
from scipy.spatial import distance

video_folder=r'D:\Python\Temp'
video='H2202_P02_SW1_SAM_overlay.mp4'

file_folder=r'D:\Python\Temp'
file_name='H2202_P02_SW1_SAM.csv'

BEH_video= os.path.join(video_folder, video)
DLC_file= os.path.join(file_folder, file_name)

output_folder=r'D:\Python\Temp\out'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
def sandwell_loc(behavioural_video):
    """Identify Sandwell locations and radii from first frame

    INPUT
    - Behavioural video
    
    OUTPUT
    - Sandwell locations for: sw1, sw2, sw3"""
    
    # Read in first video frame
    cap = cv2.VideoCapture(behavioural_video)
    correct_sandwell = 'n'
    frame=1
    while correct_sandwell != 'y':
        img = cap.read()[1] # Read in Pixel values
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_blurred = cv2.blur(gray, (3, 3)) 

        # Circle detection algorithm
        sandwells = cv2.HoughCircles(gray_blurred,  
                           cv2.HOUGH_GRADIENT, 1, 100, param1 = 50, 
                       param2 = 30, minRadius = 10, maxRadius = 20)
        if sandwells is not None:
            # Convert the circle parameters a, b and r to integers. 
            sandwells = np.uint16(np.around(sandwells)) 
        
        # Manually check that sandwells are correct
        if frame == 1 or frame % 100 == 0:
            for pt in sandwells[0, :]: 
                x, y, r = pt[0], pt[1], pt[2]
                cv2.circle(img, (x, y), r+20, (0, 255, 0), 2)
                
            avg_dis = np.mean(distance.cdist(sandwells[0][:,:2], sandwells[0][:,:2], 'euclidean'))
            if len(sandwells[0])==3 and avg_dis < 130:
                get_ipython().run_line_magic('matplotlib', 'inline')
                plt.imshow(img) 
                plt.show()
                correct_sandwell = input("Are all sandwells correct - y/n?")
                frame+=1
            else:
                correct_sandwell = 'n'
                frame+=1
        else:
            frame+=1
            
        # Classify which sandwell is sw1, sw2, sw3
        for pt in sandwells[0, :]: 
            x, y = pt[0], pt[1]
            if y == min(np.array(sandwells).transpose()[1]):
                sw1 = [float(x),float(y)]
            elif x == max(np.array(sandwells).transpose()[0]):
                sw2 = [float(x),float(y)]
            else:
                sw3 = [float(x),float(y)]
    print(len(sandwells[0]),'Wells correctly detected')
    return sw1,sw2,sw3
sandwells=sandwell_loc(BEH_video)


# In[ ]:


#prints out the sandwell positions

sandwells
#a=([296.0, 92.0], [416.0, 156.0], [294.0, 288.0])


# In[ ]:


DLC_data=pd.read_csv(DLC_file, sep=',', index_col=0)
xy=DLC_data[['x','y']]

#translates xy coordinates to use SW1 as origin

X_TRANS=xy['x']-sandwells[0][0]
Y_TRANS=xy['y']-sandwells[0][1]
XY_CENTRED_ON_SW1=pd.concat([X_TRANS, Y_TRANS], axis=1, keys=['x', 'y'])

#if the difference between the x position of SW1 and SW3 is greater than 4 pixels (roughly 1 cm) the arena is considered aligned. 
#Else rotation is performed

if abs(sandwells[2][0] - sandwells[0][0]) > 3 :
    
    #calculates cos and sin value of the angle between SW3 and SW1 which is used to rotate
    
    
    COS_ALPHA= (sandwells[2][1] - sandwells[0][1])/math.sqrt((sandwells[0][1] - sandwells[2][1])**2 + (sandwells[0][0] - sandwells[2][0])**2)
    SIN_ALPHA= (sandwells[2][0] - sandwells[0][0])/math.sqrt((sandwells[0][1] - sandwells[2][1])**2 + (sandwells[0][0] - sandwells[2][0])**2)
    ROT_MATRIX = np.array([[COS_ALPHA, SIN_ALPHA], [-SIN_ALPHA, COS_ALPHA]])
    INV_ROT_MATRIX = np.array([[COS_ALPHA, -SIN_ALPHA], [SIN_ALPHA, COS_ALPHA]])
    
    X_ROT=xy.loc[:,('x')]*COS_ALPHA  + xy.loc[:,('y')]*SIN_ALPHA
    Y_ROT=xy.loc[:,('x')]*(-SIN_ALPHA)  + xy.loc[:,('y')]*COS_ALPHA
    XY_ROT=pd.concat([X_ROT, Y_ROT], axis=1, keys=['x', 'y'])
    print("Arena has been rotated by", np.degrees(np.arcsin(SIN_ALPHA)), "degrees" )
    rotation_happened=True
    
# if no rotation is needed, the new cooridnates are simply the coordinates after translation 
else :
    XY_ROT=XY_CENTRED_ON_SW1
    print("No need to rotate the arena")
    rotation_happened=False
    
#ARENA_CORNERS are the arena corners in order 0:NW, 1:NE, 2:SW, 3:SE
#same for the S Startbox  
#relative to SW1 reference point
NEW_ARENA_CORNERS=([-99.6,-27.5], [220.4,-27.5], [-99.6,292.5], [220.4,292.5])
NEW_SOUTH_STARTBOX_CORNERS=([44.0,317.4],[154.0,317.4],[44.0,417.4],[154.0,417.4])

#TRT : translation - rotation - translation

# after rotation, the coordinates can be translated using the upper right corner as origin = this is done by subrating the vector from SW1 to the corner in the new coordinate space

x_new_coordinates=XY_ROT['x']-NEW_ARENA_CORNERS[0][0]
y_new_coordinates=XY_ROT['y']-NEW_ARENA_CORNERS[0][1]
new_coordinates=pd.concat([x_new_coordinates, y_new_coordinates], axis=1, keys=['x', 'y'])
new_coordinates


# In[ ]:


#superimposes the new arena after rotation and translation to a frame of the video to check that it's correct
#this calculates the new corners in the old axis system

cap = cv2.VideoCapture(BEH_video)

if rotation_happened: 
    NW=[INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[0])[0],INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[0])[1]]
    NE=[INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[1])[0],INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[1])[1]]
    SW=[INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[2])[0],INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[2])[1]]
    SE=[INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[3])[0],INV_ROT_MATRIX.dot(NEW_ARENA_CORNERS[3])[1]]
    ARENA_CORNERS=(NW,NE,SW,SE)
    
    SB_NW=[INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[0])[0],INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[0])[1]]
    SB_NE=[INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[1])[0],INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[1])[1]]
    SB_SW=[INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[2])[0],INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[2])[1]]
    SB_SE=[INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[3])[0],INV_ROT_MATRIX.dot(NEW_SOUTH_STARTBOX_CORNERS[3])[1]]
    SOUTH_STARTBOX_CORNERS=(SB_NW,SB_NE,SB_SW,SB_SE)

else: 
    print('No rotation happened')
    ARENA_CORNERS=NEW_ARENA_CORNERS
    SOUTH_STARTBOX_CORNERS=NEW_SOUTH_STARTBOX_CORNERS

ARENA = np.array ([ [(ARENA_CORNERS[0][0] + sandwells[0][0]),(ARENA_CORNERS[0][1] + sandwells[0][1])],[(ARENA_CORNERS[1][0] + sandwells[0][0]),(ARENA_CORNERS[1][1] + sandwells[0][1])],[(ARENA_CORNERS[3][0] + sandwells[0][0]),(ARENA_CORNERS[3][1] + sandwells[0][1])],[(ARENA_CORNERS[2][0] + sandwells[0][0]),(ARENA_CORNERS[2][1] + sandwells[0][1])] ], np.int32)
ARENA = ARENA.reshape((-1,1,2))
SB = np.array ([ [(SOUTH_STARTBOX_CORNERS[0][0] + sandwells[0][0]),(SOUTH_STARTBOX_CORNERS[0][1] + sandwells[0][1])],[(SOUTH_STARTBOX_CORNERS[1][0] + sandwells[0][0]),(SOUTH_STARTBOX_CORNERS[1][1] + sandwells[0][1])],[(SOUTH_STARTBOX_CORNERS[3][0] + sandwells[0][0]),(SOUTH_STARTBOX_CORNERS[3][1] + sandwells[0][1])],[(SOUTH_STARTBOX_CORNERS[2][0] + sandwells[0][0]),(SOUTH_STARTBOX_CORNERS[2][1] + sandwells[0][1])] ], np.int32)
SB = SB.reshape((-1,1,2))

#also draws the startbox area and the area considered startbox when the animal exits

for frame in range(1):
    cap.set(1, int(frame))
    ret, img = cap.read()
    gray_blurred = cv2.blur(img, (1, 1))
    new = cv2.circle(gray_blurred, (int(sandwells[2][0]), int(sandwells[2][1])), 1, (255, 255 , 0), -1)
    new = cv2.circle(new, (int(sandwells[1][0]), int(sandwells[1][1])), 1, (255, 255, 0), -1)
    new = cv2.circle(new, (int(sandwells[0][0]), int(sandwells[0][1])), 1, (255, 255, 0), -1)
    new = cv2.circle(new, (int(SOUTH_STARTBOX_CORNERS[0][0] + sandwells[0][0]+20), int(SOUTH_STARTBOX_CORNERS[0][1] + sandwells[0][1])),40, (255, 255, 0), 0)
    new = cv2.polylines(new,[ARENA],True,(255,255,0))
    new = cv2.polylines(new,[SB],True,(255,0,0))

    name = os.path.join(r'D:\Python\Temp\out', str(video[:-4]) + str(frame) + '.jpg')
    print ('Creating...' + name)
    cv2.imwrite(name, new)
    
    plt.imshow(new)  
    plt.show()

cap.release()
cv2.destroyAllWindows()


# In[ ]:


#function to map any point in the new coordinate system given the initial x,y 

def to_new_coordinates(x, y, *sandwells):
    x_trans=x-sandwells[0][0]
    y_trans=y-sandwells[0][1]
    
    if abs(sandwells[2][0] - sandwells[0][0]) > 4 :
        COS_ALPHA= (sandwells[2][1] - sandwells[0][1])/math.sqrt((sandwells[0][1] - sandwells[2][1])**2 + (sandwells[0][0] - sandwells[2][0])**2)
        SIN_ALPHA= (sandwells[2][0] - sandwells[0][0])/math.sqrt((sandwells[0][1] - sandwells[2][1])**2 + (sandwells[0][0] - sandwells[2][0])**2)
        x_rot=x_trans*COS_ALPHA  + y_trans*SIN_ALPHA
        y_rot=x_trans*(-SIN_ALPHA)  + y_trans*COS_ALPHA
    else :
        x_rot=x_trans
        y_rot=y_trans
    
    x_new=x_rot-NEW_ARENA_CORNERS[0][0]
    y_new=y_rot-NEW_ARENA_CORNERS[0][1]
    return [x_new,y_new]


def euclidean_distance(A, B):
    dist = sqrt( (A[0]-B[0])**2 + (A[1]-B[1])**2 )
    return dist

#estracts intervals when consecutive items in a list have the same value 

def interval_extract(list): 
    list = sorted(set(list)) 
    range_start = previous_number = list[0] 
  
    for number in list[1:]: 
        if number == previous_number + 1: 
            previous_number = number 
        else: 
            yield [range_start, previous_number] 
            range_start = previous_number = number 
    yield [range_start, previous_number] 
    
#---------------------------------------------

#for each interval, create a list with the items in the list

def intervals_to_list(interval_list):
    frames=[]
    for interval in interval_list:
        frames=frames+ list(range(interval[0],(interval[1]+1)))
    return frames

#---------------------------------------------
#outbound paths (SB to SW) are defined for each interval between the last point of leaving the SB and the first point of reaching the REW sandwell
def out_paths(list, at_the_RW_SW):
    
    for interval in list:
        j=range_start= int(interval[0])
        for j in range(interval[0],interval[1]+1):
            range_end=int(interval[1])
            if j in at_the_RW_SW:
                if j <= int(interval[1]):
                    range_end=j-1
                    yield [range_start, range_end, interval, 'outbound']
                    break
            else: 
                if j == int(interval[1]):
                    yield [range_start, range_end, interval, 'nope']

#---------------------------------------------
#return paths (SW to SB) are defined for each interval between the last point of leaving the SW and the first point of reaching the SB

def return_paths(list, at_the_RW_SW):

    for interval in list:
        j=range_start= int(interval[1])
        for j in range(interval[0],interval[1]+1)[::-1]:
            range_end=int(interval[0])
            if j in at_the_RW_SW:
                if j >= int(interval[0]):
                    range_end=j+1
                    yield [ range_end, range_start, interval, 'inbound']
                    break
            else: 
                if j == int(interval[0]):
                    yield [ range_end, range_start, interval, 'nope']
  
#-----------------------------------------------


def same_format(LIST, a):
    output=[]
    for item in LIST:
        INTERVAL=[[item[0], item[1], item, '%s' %a]]
        output=output + INTERVAL
    return output

#-------------------------------------------------
#calculates the corners in the new axis system

sandwells_new_origin=(to_new_coordinates(sandwells[0][0], sandwells[0][1], *sandwells),to_new_coordinates(sandwells[1][0], sandwells[1][1], *sandwells), to_new_coordinates(sandwells[2][0], sandwells[2][1], *sandwells))
arena_new_origin= ([NEW_ARENA_CORNERS[0][0]-NEW_ARENA_CORNERS[0][0],NEW_ARENA_CORNERS[0][1]-NEW_ARENA_CORNERS[0][1]],[NEW_ARENA_CORNERS[1][0]-NEW_ARENA_CORNERS[0][0],NEW_ARENA_CORNERS[1][1]-NEW_ARENA_CORNERS[0][1]],[NEW_ARENA_CORNERS[2][0]-NEW_ARENA_CORNERS[0][0],NEW_ARENA_CORNERS[2][1]-NEW_ARENA_CORNERS[0][1]],[NEW_ARENA_CORNERS[3][0]-NEW_ARENA_CORNERS[0][0],NEW_ARENA_CORNERS[3][1]-NEW_ARENA_CORNERS[0][1]])
South_SB_new_origin= ([NEW_SOUTH_STARTBOX_CORNERS[0][0]-NEW_ARENA_CORNERS[0][0],NEW_SOUTH_STARTBOX_CORNERS[0][1]-NEW_ARENA_CORNERS[0][1]],[NEW_SOUTH_STARTBOX_CORNERS[1][0]-NEW_ARENA_CORNERS[0][0],NEW_SOUTH_STARTBOX_CORNERS[1][1]-NEW_ARENA_CORNERS[0][1]],[NEW_SOUTH_STARTBOX_CORNERS[2][0]-NEW_ARENA_CORNERS[0][0],NEW_SOUTH_STARTBOX_CORNERS[2][1]-NEW_ARENA_CORNERS[0][1]],[NEW_SOUTH_STARTBOX_CORNERS[3][0]-NEW_ARENA_CORNERS[0][0],NEW_SOUTH_STARTBOX_CORNERS[3][1]-NEW_ARENA_CORNERS[0][1]])

Radius=20
pixel_size=0.311
rewarded_sandwell=file_name[10:13]

if rewarded_sandwell=='SW1':
    RW_SW=0
elif rewarded_sandwell=='SW2':
    RW_SW=1
elif rewarded_sandwell=='SW3':
    RW_SW=2
  
    
rat_position=['startbox']

for i in new_coordinates.index[1:]:
    new_xy=[new_coordinates['x'][i], new_coordinates['y'][i]]
    if South_SB_new_origin[0][0] <= new_coordinates['x'][i] and new_coordinates['x'][i] <= South_SB_new_origin[1][0] and South_SB_new_origin[0][1] <= new_coordinates['y'][i] and new_coordinates['y'][i] <= South_SB_new_origin[2][1]:
        rat_position.append('startbox')
    elif euclidean_distance(new_xy, [South_SB_new_origin[0][0]+20, South_SB_new_origin[0][1]]) < 40 :
        if rat_position[i-1] == 'startbox':
            rat_position.append ('startbox')
        else:
            rat_position.append('arena')
    else :
            rat_position.append('arena')
            

df=pd.DataFrame(data={'c':DLC_data['c'], 'x': new_coordinates['x'], 'y': new_coordinates['y'], 'x_cm': new_coordinates['x']*pixel_size, 'y_cm': new_coordinates['y']*pixel_size, 'where':rat_position, 'Speed':DLC_data['Speed'], 'Speed_smooth':DLC_data['Speed_smooth']})
df=df[['c','x', 'y', 'x_cm', 'y_cm','Speed', 'Speed_smooth','where']]
        
#intervals of frames when the animal is in the SB or out of it


in_the_SB=list(interval_extract(df[df['where']=='startbox'].index))
out_the_SB=list(interval_extract(df[df['where']!='startbox'].index))

#--------------------------------

at_the_RW_SW=[]

for i in new_coordinates.index:
    new_xy=[new_coordinates['x'][i], new_coordinates['y'][i]]
    if euclidean_distance(new_xy, sandwells_new_origin[RW_SW]) < 20:
        at_the_RW_SW.append(i)
        
at_the_RW_SW_list=same_format(list(interval_extract(at_the_RW_SW)), "rewarded_SW")
in_the_SB_list=same_format(in_the_SB, "startbox") 
outbound_list=(list(out_paths(out_the_SB,at_the_RW_SW)))
inbound_list=(list(return_paths(out_the_SB,at_the_RW_SW)))

#---------------------------------
print(at_the_RW_SW_list)
print(in_the_SB_list)
print(outbound_list)
print(inbound_list)
#---------------------------------


path = []
path_index=[]

group = in_the_SB_list + outbound_list + inbound_list + at_the_RW_SW_list

group= [i for i in group if i[3] != "nope"]


all=[]

df3 = pd.DataFrame({'1': new_coordinates.index})

for i in group: 
    all = all + [i[2]]

#creates a df with index and value (and nothing elsewhere. these will converted to NaN when merged)
    
for item in group:
    for j in range(item[0], item[1]+1):
        path.append(item[3])
        path_index.append(j)

df4 = pd.DataFrame(path, index =path_index, columns =['direction'])

result = pd.concat([df, df4], axis=1)

result[1800:9850]


# In[ ]:


group


# In[ ]:


result[result['direction'] == "inbound"].plot.scatter('x', 'y', c ='cyan', s=2)
XY_ROT.plot.scatter('x', 'y', c ='c', s=2)
NEW_ARENA_CORNERS


# In[ ]:


arena_new_origin


# In[ ]:


result.to_csv(os.path.join(output_folder, video[:-4] + '_translated.csv'), sep='\t')


# In[ ]:




