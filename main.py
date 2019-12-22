import cv2
import numpy as np
import time
import screeninfo
import random
import yoloface
import pandas as pd
cam = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cascPath = "face.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
df = pd.read_csv("setting.csv")
img_counter = 0
n_delay = int(df["Delay"].values[0])

max_images = int(df["MaxImages"].values[0])

str_shot = df["ShootSize"].values[0].replace("(","")
str_shot = str_shot.replace(")","")
shot_width = str_shot.split(",")[0]
shot_height = str_shot.split(",")[1]

shoot_size = (int(shot_width),int(shot_height))

str_BG = df["BackGround"].values[0].replace("(","")
str_BG = str_BG.replace(")","")
R = str_BG.split(",")[0]
G = str_BG.split(",")[1]
B = str_BG.split(",")[2]
BG = (R,G,B)
shot_images = []
x_cor = []
y_cor = []
z_index = []
cur_index = 0
cur_time = time.time()
# get the size of the screen
screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

# blank_img = np.ones((1000,1200),dtype=np.uint8)
refPt = None
refCurRect = None
moving = False
cur_face_frame = np.ones((shoot_size[1],shoot_size[0],3),dtype=np.uint8)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)

cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

blank_img = np.ones((height,width,3),dtype=np.uint8)
def click_and_release(event, x, y, flags, param):
	# # grab references to the global variables
    global refPt, moving,refCurRect,shot_images,x_cor,y_cor,z_index
    blank_img = np.ones((height,width),dtype=np.uint8)
    if(len(x_cor)>0):
        pass
    else:
        return
    # print(x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        for index in reversed(z_index):
            if(x>x_cor[index] and x<(x_cor[index] + shot_images[index].shape[1]) and y>y_cor[index] and y<(y_cor[index]+shot_images[index].shape[0])):
                moving = True
                if(index == (cur_index-1)):
                    pass
                else:
                    tem_x = x_cor[index]
                    tem_y = y_cor[index]
                    tem_img = shot_images[index]
                    for n_i in range(index,cur_index):
                        if(n_i == (cur_index-1)):
                            x_cor[n_i] = tem_x
                            y_cor[n_i] = tem_y
                            shot_images[n_i] = tem_img 
                            break
                        x_cor[n_i] = x_cor[n_i+1]
                        y_cor[n_i] = y_cor[n_i+1]
                        shot_images[n_i] = shot_images[n_i+1]
                        pass
                refPt = (x,y)
                break
    elif event == cv2.EVENT_LBUTTONUP:
        refPt = (x,y)
        moving = False
    if(moving):
        dx = x - refPt[0]
        dy = y - refPt[1]
        new_x = x_cor[cur_index-1] + dx
        new_y = y_cor[cur_index-1] + dy
        # refCurRect = (new_x,new_y,refCurRect[2],refCurRect[3])
        # show_x_y(shot_images[0],new_x,new_y)
        x_cor[cur_index-1] = new_x
        y_cor[cur_index-1] = new_y
        show_result(shot_images,x_cor,y_cor,z_index)
        refPt = (x,y)
        pass

cv2.setMouseCallback("window", click_and_release)

def show_x_y(image,x,y,blank_img):
    img = image.copy()
    # blank_img = np.ones((height,width),dtype=np.uint8)
    max_x = blank_img.shape[1]-image.shape[1]
    max_y = blank_img.shape[0]-image.shape[0]
    if(x>max_x):
        x = max_x
    if(x<0):
        x = 0
    if(y<0):
        y = 0
    if(y>max_y):
        y = max_y
    global refCurRect
    refCurRect = (x,y,img.shape[1],img.shape[0])
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    blank_img[y:img.shape[0]+y,x:img.shape[1]+x] = img
    cv2.imshow("window", blank_img)
    pass

def show_result(images,x_cor,y_cor,z_index):
    global blank_img
    # blank_img[:] = BG
    blank_img.fill(255)
    for index in z_index:
        img = images[index].copy()
        x = x_cor[index]
        y = y_cor[index]
        show_x_y(img,x,y,blank_img)
    pass

def shoot_image(image,delay = 0,max_image_count = max_images):
    img = image.copy()
    global cur_time,shot_images,x_cor,y_cor,width,height,cur_index,z_index
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,shoot_size,interpolation=cv2.INTER_AREA)
    cur_face_frame = img
    if((time.time()-cur_time)>delay ):
        if(len(shot_images)>(max_image_count-1)):
            # shot_images.pop()
            # shot_images.append(img)
            for n_index in range(len(z_index)):
                if(n_index == (len(z_index)-1)):
                    x_cor[n_index] = random.randint(0,width-shoot_size[0])
                    y_cor[n_index] = random.randint(0,height-shoot_size[1])
                    shot_images[n_index] = img.copy()
                    # z_index[n_index] = n_index
                    # print(n_index)
                else:
                    shot_images[n_index] = shot_images[n_index+1].copy()
                    x_cor[n_index] = x_cor[n_index+1]
                    y_cor[n_index]  = y_cor[n_index+1]
                    # z_index[n_index] = z_index[n_index+1]
            # print(shot_images[0][100])
            show_result(shot_images,x_cor,y_cor,z_index)
            return
            pass
        x = random.randint(0,width-shoot_size[0])
        y = random.randint(0,height-shoot_size[1])
        z_index.append(cur_index)
        x_cor.append(x)
        y_cor.append(y)
        shot_images.append(img)
        cur_time = time.time()
        cur_index+=1
        show_result(shot_images,x_cor,y_cor,z_index)
    else:
        pass
def image_diff(left,right):
    # the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((left.astype("float") - right.astype("float")) ** 2)
	err /= float(left.shape[0] * left.shape[1])
	return err
frames = []
rect = None
frame_stack = 0
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, weights = hog.detectMultiScale(gray)
    max_n = 0
    result_img = None
    if((time.time()-cur_time)>n_delay):
        cur_time = time.time()
        for image in frames:
            rects = yoloface.get_facebyyolo(image.copy())
            if(len(rects)>0):
                # detect face
                diff = image_diff(cur_face_frame,image)
                if(diff>max_n):
                    max_n = diff
                    result_img = image
                    rect = rects[0]
                    break
                pass
            pass
        if(result_img is not None):
            shoot_image(result_img.copy())
            cur_face_frame = result_img
        frames.clear()
        pass
    else:
        frame_stack = frame_stack+1
        if(frame_stack%5 ==0):
            frame_stack = 0
            image = cv2.resize(frame.copy(),shoot_size)
            frames.append(image)
        pass
    cv2.imshow("camera", frame)
    if not ret:
        break
    k = cv2.waitKey(10)

cam.release()

cv2.destroyAllWindows()