import random
import math 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Function to quantize depth image to n depth levels.
def quantize(depth_image, number_to_divide):
  image = np.zeros(depth_image.shape)
  for i in range(depth_image.shape[0]):
    for j in range(depth_image.shape[1]):
      image[i][j] = int(depth_image[i][j]/number_to_divide)
  return image

#Function for calculating pocs and drawing matches.
def point_correspondences(image1, image2):
  orb = cv.ORB_create()
  #Obtain keypoints and descriptors.
  keypoints1, des1 = orb.detectAndCompute(image1,None)
  keypoints2, des2 = orb.detectAndCompute(image2,None)
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  matches = sorted(matches, key= lambda x:x.distance)
  image3 = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:20], None, flags=2)
  #Draw macthes.
  plt.figure()
  plt.imshow(image3)
  plt.show()
  cv.imwrite("Matches.jpg", image3)
 
  #To store point correspondences.
  poc = []
  for elem in matches:
    im_1_index = (elem.queryIdx)
    im_2_index = (elem.trainIdx)
    x_1 = keypoints1[im_1_index].pt
    x_1_ = keypoints2[im_2_index].pt
    poc.append((x_1,x_1_))
  
  #To store respective points of image1 and image2 present in the poc.
  points1 = []
  points2 = []

  for elem in matches:
    image1_ = elem.queryIdx
    image2_ = elem.trainIdx
    (x1,y1) = keypoints1[image1_].pt
    (x2,y2) = keypoints2[image2_].pt
    points1.append((x1, y1))
    points2.append((x2, y2))

  points1 = np.asarray(points1)
  points2 = np.asarray(points2)

  return poc, points1, points2

#Function to calculate homography.
def calculateHomography(poc):
  columns = 9
  rows = len(poc) * 2
  A = np.zeros((rows, columns))

  for i in range(len(poc)):
    x_1 = (poc[i][0][0])
    y_1 = (poc[i][0][1])
    z_1 = 1
    x_2 = (poc[i][1][0])
    y_2 = (poc[i][1][1])
    z_2 = 1
    A[2*i, :] = [x_1, y_1, 1, 0, 0, 0, -x_2*x_1, -x_2*y_1, -x_2*z_1]
    A[2*i+1, :] = [0, 0, 0, x_1, y_1, 1, -y_2*x_1, -y_2*y_1, -y_2*z_1]
	
  #SVD.
  U, D, V_t = np.linalg.svd(A)
  hom = V_t[-1]
  HomographyMatrix = np.zeros((3,3))
  HomographyMatrix[0] = hom[:3]
  HomographyMatrix[1] = hom[3:6]
  HomographyMatrix[2] = hom[6:9]
  HomographyMatrix = HomographyMatrix / HomographyMatrix[2,2]

  return HomographyMatrix

#Implementation of Ransac Algorithm.
def ransac(poc):
  best_H = 0
  best_so_far = 0

  for i in range(10000):
    inliers = []
    index = random.sample(range(1,len(poc)),4)
    matches = [poc[i] for i in index]
    H = calculateHomography(matches)
    threshold = 10
    number_inliers = 0

    for elem in range(len(poc)):
      curr = poc[elem][0]
      new_elem = np.asarray([curr[0],curr[1],1]).T
      transformed = H.dot(new_elem)
      transformed = transformed / transformed[2]
      act = np.asarray([poc[elem][1][0], poc[elem][1][1],1])
      if np.linalg.norm(transformed-act) <= threshold:
        number_inliers+=1
        inliers.append(poc[elem])
    
    if number_inliers > best_so_far:
      best_so_far = number_inliers
      best_H = calculateHomography(inliers)
   
  return best_H

#Function to warp image using depth.
def warpimage_depth(image1, image2, depth_image_final, x_offset, y_offset, x, y, H, lev):
  ref_image = image1
  src_image = image2		
  img_temp = np.zeros((x,y,3))

  for elem in range(lev):
    for i in range(ref_image.shape[0]):
      for j in range(src_image.shape[1]):
        #If depth level equals the value at the corresponding pixel in depth image, use the homography corresponding to that depth level.
        if H[elem][0] == depth_image_final[i][j]:
          h_dot = H[elem][1].dot(np.asarray([j,i,1]).T)
          h_dot = h_dot/h_dot[2]
          h_dot[0] = int(h_dot[0])
          h_dot[1] = int(h_dot[1])
          x_c = int(h_dot[0])
          y_c = int(h_dot[1])
          try:
              for i1 in range(-2,3):  #For continuity of pixels.
                for i2 in range(-2,3):
                    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = src_image[i,j]
          except:
            continue
        #If depth level doesnot match the value at the corresponding pixel in depth image, use the homography corresponding to the first depth level.
        else:
          h_dot = H[0][1].dot(np.asarray([j,i,1]).T)
          h_dot = h_dot/h_dot[2]
          h_dot[0] = int(h_dot[0])
          h_dot[1] = int(h_dot[1])
          x_c = int(h_dot[0])
          y_c = int(h_dot[1])
          try:
              for i1 in range(-2,3):
                for i2 in range(-2,3):
                    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = src_image[i,j]
          except:
            continue

  img_temp = img_temp.astype(np.uint8)
  return img_temp

#Function to warp image without using depth.
def warpimage(image1, image2, x_offset, y_offset, x, y, H, lev):
  ref_image = image1
  src_image = image2		
  img_temp = np.zeros((x,y,3))

  for elem in range(lev):
    for i in range(ref_image.shape[0]):
      for j in range(src_image.shape[1]):
          h_dot = H[0][1].dot(np.asarray([j,i,1]).T)
          h_dot = h_dot/h_dot[2]
          h_dot[0] = int(h_dot[0])
          h_dot[1] = int(h_dot[1])
          x_c = int(h_dot[0])
          y_c = int(h_dot[1])
          try:
              for i1 in range(-2,3):    #For obtaining continuity in final image.
                for i2 in range(-2,3):
                    img_temp[y_c+x_offset+i1][x_c+y_offset+i2] = src_image[i,j]
          except:
            continue

  img_temp = img_temp.astype(np.uint8)
  return img_temp

#Number of depth levels to divide to.
depth_levels = 11
number_to_divide = math.ceil( 256 / depth_levels )

#2nd Folder.
ref_image = cv.imread('/content/drive/MyDrive/RGBD dataset/000000292/im_0.jpg')
src_image = cv.imread('/content/drive/MyDrive/RGBD dataset/000000292/im_1.jpg')
depth_image_src = cv.imread('/content/drive/MyDrive/RGBD dataset/000000292/depth_0.jpg')
depth_image_src = cv.cvtColor(depth_image_src, cv.COLOR_BGR2GRAY)
#Quantising.
depth_image_final = quantize(depth_image_src, number_to_divide)

hom_list = [[] for i in range(depth_levels)]

#Finding point correspondences.
poc, points1, points2 = point_correspondences(ref_image, src_image)

#Finding poc for each depth level.
for elem in range(len(poc)):
		for i in range(depth_image_src.shape[0]):
			for j in range(depth_image_src.shape[1]):
				if points2[elem][0]==i and points2[elem][1]==j:
					hom_list[int(depth_image_src[i][j]//number_to_divide)].append(poc[elem])
     
#Question e
#Using own homography.
print("Warped image using own homography (with depth).\n")
H = []
best_level = -1

for i in range(depth_levels):
    if len(hom_list[i]) > 4:
	      best_level = max(best_level, i)
	      H.append((i,ransac(np.asarray(hom_list[i]))))       
    else:
        H.append((i,H[best_level][1]))

lev = depth_levels
image = warpimage_depth(ref_image, src_image, depth_image_final, 500,500,1500,1500,H,lev)
cv.imwrite("2nd-own homography (with depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Using inbuilt homography.
print("Warped image using inbuilt homography (with depth).\n")
H_actual = []
points_list1_best = []
points_list2_best = []
for i in range(depth_levels):
    if len(hom_list[i]) > 4:   
        best_level = max(best_level, i)   
        points_list1 = []
        points_list2 = []
        for items in hom_list[i]:
          if len(items) > 0:
            points_list1.append(items[0])
            points_list2.append(items[1])
        points_list1_best = points_list1
        points_list2_best = points_list2
        H,_ = cv.findHomography(np.array(points_list1), np.array(points_list2), cv.RANSAC, 100)
        H_actual.append((i,H))
    else:
        H, _ = cv.findHomography(np.array(points_list1_best), np.array(points_list2_best), cv.RANSAC, 100)
        H_actual.append((i,H))

lev = depth_levels
image = warpimage_depth(ref_image, src_image, depth_image_final, 500,500,1500,1500,H_actual,lev)
cv.imwrite("2nd-inbuilt homography (with depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Question f
#Using own homography.
print("Warped image using own homography (without depth).\n")
H = []
H_hat = ransac(poc)
H.append((0,H_hat))
image = warpimage(ref_image, src_image, 500, 500, 1500, 1500, H, 1)
cv.imwrite("2nd-own homography (without depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Using In-built homography.
print("Warped image using inbuilt homography (without depth).\n")
H = []
H_hat, _ = cv.findHomography(points1, points2, cv.RANSAC, 100)
H.append((0,H_hat))
image = warpimage(ref_image, src_image, 200, 200, 800, 800, H, 1)
cv.imwrite("2nd-inbuilt homography (without depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()


#5th Folder.
ref_image = cv.imread('/content/drive/MyDrive/RGBD dataset/000002812/im_1.jpg')
src_image = cv.imread('/content/drive/MyDrive/RGBD dataset/000002812/im_2.jpg')
depth_image_src = cv.imread('/content/drive/MyDrive/RGBD dataset/000002812/depth_2.jpg')
depth_image_src = cv.cvtColor(depth_image_src, cv.COLOR_BGR2GRAY)
#Quantising.
depth_image_final = quantize(depth_image_src, number_to_divide)

#Finding point correspondences.
poc, points1, points2 = point_correspondences(ref_image, src_image)
hom_list = [[] for i in range(depth_levels)]

#Finding poc for each depth level.
for elem in range(len(poc)):
		for i in range(depth_image_src.shape[0]):
			for j in range(depth_image_src.shape[1]):
				if points2[elem][0]==i and points2[elem][1]==j:
					hom_list[int(depth_image_src[i][j]//number_to_divide)].append(poc[elem])

#Question e
#Using own homography.
print("Warped image using own homography (with depth).\n")
H = []
best_level = -1

for i in range(depth_levels):

    if len(hom_list[i])>4:
	      best_level = max(best_level, i)
	      H.append((i,ransac(np.asarray(hom_list[i]))))
       
    else:
        H.append((i,H[best_level][1]))

lev=9
image = warpimage_depth(ref_image, src_image, depth_image_final, 500,500,1200,1200,H,lev)
cv.imwrite("5th-own homography (with depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Using inbuilt homography.
print("Warped image using inbuilt homography (with depth).\n")
H_actual = []
points_list1_best = []
points_list2_best = []
for i in range(depth_levels):
    if len(hom_list[i]) > 4:   
        best_level = max(best_level, i)   
        points_list1 = []
        points_list2 = []
        for items in hom_list[i]:
          if len(items) > 0:
            points_list1.append(items[0])
            points_list2.append(items[1])
        points_list1_best = points_list1
        points_list2_best = points_list2
        H,_ = cv.findHomography(np.array(points_list1), np.array(points_list2), cv.RANSAC, 10000)
        H_actual.append((i,H))
    else:
        H, _ = cv.findHomography(np.array(points_list1_best), np.array(points_list2_best), cv.RANSAC, 10000)
        H_actual.append((i,H))

lev=9
image = warpimage_depth(ref_image, src_image, depth_image_final, 500,500,1200,1200,H_actual,lev)
cv.imwrite("5th-inbuilt homography (with depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Question f
#Using own homography.
print("Warped image using own homography (without depth).\n")
H = []
H_hat = ransac(poc)
H.append((0,H_hat))
image = warpimage(ref_image, src_image, 500, 500, 1200, 1200, H, 1)
cv.imwrite("5th-own homography (without depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Using In-built homography.
print("Warped image using inbuilt homography (without depth).\n")
H = []
H_hat, _ = cv.findHomography(points1, points2, cv.RANSAC, 100)
H.append((0,H_hat))
image = warpimage(ref_image, src_image, 500, 500, 1200, 1200, H, 1)
cv.imwrite("5th-inbuilt homography (without depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()


#3rd Folder.
ref_image = cv.imread('/content/drive/MyDrive/RGBD dataset/000000705/im_2.jpg')
src_image = cv.imread('/content/drive/MyDrive/RGBD dataset/000000705/im_3.jpg')
depth_image_src = cv.imread('/content/drive/MyDrive/RGBD dataset/000000705/depth_3.jpg')
depth_image_src = cv.cvtColor(depth_image_src, cv.COLOR_BGR2GRAY)
#Quantising.
depth_image_final = quantize(depth_image_src, number_to_divide)

hom_list = [[] for i in range(depth_levels)]

#Finding point correspondences.
poc, points1, points2 = point_correspondences(ref_image, src_image)

#Finding poc for each depth level.
for elem in range(len(poc)):
		for i in range(depth_image_src.shape[0]):
			for j in range(depth_image_src.shape[1]):
				if points2[elem][0]==i and points2[elem][1]==j:
					hom_list[int(depth_image_src[i][j]//number_to_divide)].append(poc[elem])
     
#Question e
#Using own homography.
print("Warped image using own homography (with depth).\n")
H = []
best_level = -1

for i in range(depth_levels):
    if len(hom_list[i]) > 4:
	      best_level = max(best_level, i)
	      H.append((i,ransac(np.asarray(hom_list[i]))))       
    else:
        H.append((i,H[best_level][1]))

lev = depth_levels
image = warpimage_depth(ref_image, src_image, depth_image_final, 500,500,1500,1500,H,lev)
cv.imwrite("3rd-own homography (with depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Using inbuilt homography.
print("Warped image using inbuilt homography (with depth).\n")
H_actual = []
points_list1_best = []
points_list2_best = []
for i in range(depth_levels):
    if len(hom_list[i]) > 4:   
        best_level = max(best_level, i)   
        points_list1 = []
        points_list2 = []
        for items in hom_list[i]:
          if len(items) > 0:
            points_list1.append(items[0])
            points_list2.append(items[1])
        points_list1_best = points_list1
        points_list2_best = points_list2
        H,_ = cv.findHomography(np.array(points_list1), np.array(points_list2), cv.RANSAC, 100)
        H_actual.append((i,H))
    else:
        H, _ = cv.findHomography(np.array(points_list1_best), np.array(points_list2_best), cv.RANSAC, 100)
        H_actual.append((i,H))

lev = depth_levels
image = warpimage_depth(ref_image, src_image, depth_image_final, 500,500,1500,1500,H_actual,lev)
cv.imwrite("3rd-inbuilt homography (with depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Question f
#Using own homography.
print("Warped image using own homography (without depth).\n")
H = []
H_hat = ransac(poc)
H.append((0,H_hat))
image = warpimage(ref_image, src_image, 500, 500, 1500, 1500, H, 1)
cv.imwrite("3rd-own homography (without depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()

#Using In-built homography.
print("Warped image using inbuilt homography (without depth).\n")
H = []
H_hat, _ = cv.findHomography(points1, points2, cv.RANSAC, 100)
H.append((0,H_hat))
image = warpimage(ref_image, src_image, 200, 200, 800, 800, H, 1)
cv.imwrite("3rd-inbuilt homography (without depth).jpg",image)
plt.figure()
plt.imshow(image)
plt.show()
