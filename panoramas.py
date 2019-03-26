import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def createMask(im):
    mask = np.zeros((im.shape[0], im.shape[1]))
    mask[0,:] = 1
    mask[-1,:] = 1
    mask[:,0] = 1
    mask[:,-1] = 1
    mask = distance_transform_edt(1-mask)
    # print(mask, "original")
    # mask = mask/mask.max(0)
    # buxprint(mask, "normalized")
    return mask


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...  
    warp_im1 = cv2.warpPerspective(im1, np.eye(3), (1710, 660))
    warp_im2 = cv2.warpPerspective(im2, H2to1, (1710, 660))
    cv2.imwrite('../results/6_1.jpg', warp_im2)
    
    im2_mask = createMask(im2)
    im2_mask = np.stack((im2_mask, im2_mask, im2_mask), axis = 2)
    im1_mask = createMask(im1)
    im1_mask = np.stack((im1_mask, im1_mask, im1_mask), axis = 2)
	
    im2_mask = cv2.warpPerspective(im2_mask, H2to1, (1710, 660))
    im1_mask = cv2.warpPerspective(im1_mask, np.eye(3), (1710, 660))
    masked_im1 = np.multiply(warp_im1, im1_mask)
    masked_im2 = np.multiply(warp_im2, im2_mask)
    # print(im1_mask)
    # print(im2_mask)
    
    pano_im = warp_im1
    for i in range(warp_im1.shape[0]):
        for j in range(warp_im1.shape[1]):
            for k in range(warp_im1.shape[2]):
                if pano_im[i, j, k] * im2_mask[i, j, k] >0:
                    pano_im[i, j, k] =  (masked_im1[i, j, k] + masked_im2[i, j, k])/(im1_mask[i, j, k] + im2_mask[i, j, k])
                else:
                    pano_im[i, j, k] = max(warp_im1[i, j, k], warp_im2[i, j, k])
    cv2.imshow('Final 6_1 panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    max_height = max(im1.shape[0], im2.shape[0])
    max_width = max(im1.shape[1], im2.shape[1])

    max_size = np.array([np.array([0, 0, 1]), np.array([0, max_height - 1, 1]),
                           np.array([max_width - 1, 0, 1]), np.array([max_width - 1, max_height - 1, 1])])
    max_size_warp = np.dot(H2to1, np.transpose(max_size))
    max_size_warp = max_size_warp / max_size_warp[2, :]
    warp_height_min = int(np.min(max_size_warp[1, :]))

    M = np.float32([[1, 0, 0], [0, 1, -warp_height_min], [0, 0, 1]])

    out_width = int(np.max(max_size_warp[0, :]))
    out_height = int(np.max(max_size_warp[1, :])) - warp_height_min
    outsize = (out_width, out_height)

    warp_im1 = cv2.warpPerspective(im1, M, outsize)
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), outsize)
    
    im2_mask = createMask(im2)
    im2_mask = np.stack((im2_mask, im2_mask, im2_mask), axis = 2)
    im1_mask = createMask(im1)
    im1_mask = np.stack((im1_mask, im1_mask, im1_mask), axis = 2)
    
    im2_mask = cv2.warpPerspective(im2_mask, np.matmul(M, H2to1), outsize)
    im1_mask = cv2.warpPerspective(im1_mask, M, outsize)
    masked_im1 = np.multiply(warp_im1, im1_mask)
    masked_im2 = np.multiply(warp_im2, im2_mask)
    # print(im1_mask)
    # print(im2_mask)
    
    pano_im = warp_im1
    for i in range(warp_im1.shape[0]):
        for j in range(warp_im1.shape[1]):
            for k in range(warp_im1.shape[2]):
                if pano_im[i, j, k] * im2_mask[i, j, k] >0:
                    pano_im[i, j, k] =  (masked_im1[i, j, k] + masked_im2[i, j, k])/(im1_mask[i, j, k] + im2_mask[i, j, k])
                else:
                    pano_im[i, j, k] = max(warp_im1[i, j, k], warp_im2[i, j, k])
    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    return pano_im

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    # print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1.npy', H2to1)
    
    pano_im_6_1 = imageStitching(im1, im2, H2to1)
    # cv2.imwrite('../results/6_1_pan.jpg', pano_im_6_1)
    # cv2.imshow('panoramas', pano_im_6_1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    cv2.imshow('6_2 Panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    final_pano_im = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', final_pano_im)
    cv2.imshow('6_3 panoramas', final_pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()