import numpy as np
import cv2
import matplotlib



def panorama(img_left,img_right,feathering=1,bad_pic=0):
    #function takes inputs of left and right image, feathering true/false, and bad_pic true/false
    #if feathering is true, uses feathering blending, otherwise uses simple alpha belnding
    #if bad pic is true, weights the matches above a certain point (2/3 way up the picture) as higher to avoid the stuff in the foreground

    #ratio is the percent of matches to keep. We only need 5 (threshold+1) for the homographic transform, so I keep it high
    ratio = 0.90
    threshold =4

    sift = cv2.xfeatures2d.SIFT_create()

    #openCV function detectandCompute finds the keypoints in each image
    kp_l, dst_l = sift.detectAndCompute(img_left,None)
    kp_r, dst_r = sift.detectAndCompute(img_right,None)

    #Use Flann matching algorithm
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(dst_l, dst_r, k=2)

    if bad_pic==1:
        foreground =1.5

    #check to see which matches meet our criteria and discard the rest
    bestmatches = []
    if bad_pic == 1:
        for m, n in matches:
            if m.distance < ratio * n.distance and kp_l[m.queryIdx].pt[0] < img_left.shape[0]/foreground:
                bestmatches.append(m)
    else:
         for m, n in matches:
            if m.distance < ratio * n.distance:
                bestmatches.append(m)

    #For the matched key points, use findHomography to create the transformation matrix
    if len(bestmatches) > threshold:
        lpoints = np.float32([kp_l[m.queryIdx].pt for m in bestmatches]).reshape(-1, 1, 2)
        rpoints = np.float32([kp_r[m.trainIdx].pt for m in bestmatches]).reshape(-1, 1, 2)

        (M, st) = cv2.findHomography(rpoints,lpoints, cv2.RANSAC,threshold)

    #Calculate where the four corners of the transformed image are
    ltcorner = np.dot(M,[[0],[0],[1]])
    ltcorner = ltcorner/ltcorner[2]
    rtcorner = np.dot(M,[[img_right.shape[0]],[0],[1]])
    rtcorner = rtcorner/rtcorner[2]
    rbcorner = np.dot(M,[[img_right.shape[0]],[img_right.shape[1]],[1]])
    rbcorner = rbcorner/rbcorner[2]
    lbcorner = np.dot(M,[[0],[img_right.shape[1]],[1]])
    lbcorner = lbcorner/lbcorner[2]

    right = int(np.maximum(rtcorner[0],rbcorner[0]))
    top = int(np.minimum(rtcorner[1],ltcorner[1]))
    bottom = int(np.maximum(rbcorner[1],lbcorner[1]))
    left = int(np.minimum(ltcorner[0],lbcorner[0]))

    #transform the right image
    output = cv2.warpPerspective(img_right,M,(img_left.shape[1]+img_right.shape[1],img_left.shape[0]+img_right.shape[0]))
    output_blend = output.copy()

    # Blend the left and right images
    for x in range(left,img_left.shape[1],1):
        c = (np.float32(x-left))/ np.float32(img_left.shape[1]-left)
        for y in range(0,img_left.shape[0],1):
            if feathering == 1:
                b = (np.float32(y)) / np.float32(img_left.shape[0])
                a = np.minimum(c,b)
            else:
                a = c
            for z in range(0,3,1):
                if output[y,x,z] != 0:
                    output_blend[y,x,z] = int(a*np.float32(output[y,x,z])+(1-a)*np.float32(img_left[y,x,z]))
                else:
                    output_blend[y,x,z] = img_left[y,x,z]

    output[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
    output_blend[0:img_left.shape[0],0:left] = img_left[0:img_left.shape[0],0:left]

    #crop out all black space
    width=img_right.shape[1]+right
    output_crop = output[np.maximum(top,0):np.minimum(bottom,img_left.shape[0]),0:width]
    output_blend = output_blend[np.maximum(top, 0):np.minimum(bottom, img_left.shape[0]), 0:width]


    return output,output_crop, output_blend

img_left_file=raw_input('Left image filename:  ')
img_right_file=raw_input('Right image filename:  ')
img_left = cv2.imread(img_left_file)
img_right = cv2.imread(img_right_file)
img_out, img_out_crop, img_out_blend = panorama(img_left,img_right,0,0)

cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)
cv2.imshow('Panorama',img_out)
cv2.namedWindow('Panorama blend', cv2.WINDOW_NORMAL)
cv2.imshow('Panorama blend',img_out_blend)
#cv2.namedWindow('Panorama crop', cv2.WINDOW_NORMAL)
#cv2.imshow('Panorama crop',img_out_crop)
#cv2.imwrite('goodpic.jpg',img_out_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

