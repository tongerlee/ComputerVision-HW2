import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    split_pyramid = np.dsplit(gaussian_pyramid, gaussian_pyramid.shape[2])
    for i in range(len(DoG_levels)):
        DoG_pyramid.append(np.subtract(split_pyramid[i+1], split_pyramid[i]))
    DoG_pyramid = np.dstack(DoG_pyramid)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    principal_curvature = np.zeros((DoG_pyramid.shape[0], \
                                 DoG_pyramid.shape[1], DoG_pyramid.shape[2]))
    for i in range(DoG_pyramid.shape[-1]):
        grad_x = cv2.Sobel(DoG_pyramid[:, :, i], cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(DoG_pyramid[:, :, i], cv2.CV_64F, 0, 1, ksize=5)
        x_grad_x = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=5)
        y_grad_x = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=5)
        x_grad_y = cv2.Sobel(grad_y, cv2.CV_64F, 1, 0, ksize=5)
        y_grad_y = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=5)
        for x in range(grad_x.shape[0]):
            for y in range(grad_x.shape[1]):
                hessian = np.matrix([[x_grad_x[x, y], y_grad_x[x, y]], [x_grad_y[x, y], y_grad_y[x, y]]])
                if(np.linalg.det(hessian) == 0):
                    principal_curvature[x, y, i] = 0
                else:
                    principal_curvature[x, y, i]=(np.trace(hessian))**2 / np.linalg.det(hessian)
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    ##############
    #  TO DO ...
    # Compute locsDoG here
    height = DoG_pyramid.shape[0]
    width = DoG_pyramid.shape[1]
    level = DoG_pyramid.shape[2]
    for i in range(1, height):
        for j in range(1, width):
            for k in range(1, level):
                currPixel = DoG_pyramid[i, j, k]
                if abs(currPixel) > th_contrast:
                    if principal_curvature[i, j, k] < th_r:
                        neighbors =  DoG_pyramid[(i - 1): (i + 2), \
                               (j - 1): (j + 2), k].flatten()
                        neighbors = np.concatenate([neighbors,\
                                                   DoG_pyramid[i, j, \
                           (k - 1):(k + 2)]])
                        if np.max(neighbors) == currPixel or np.min(neighbors) == currPixel:
                            locsDoG.append([j, i, k])   
    return np.array(locsDoG)
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
   # im = cv2.resize(im, (im.shape[1]*6, im.shape[0]*6))
   # for loc in locsDoG:
   #     cv2.circle(im, (loc[0]*6, loc[1]*6), 4, (0, 255, 0), -1)

   # cv2.imshow('image', im)
   # cv2.waitKey(0)  # press any key to exit
   # cv2.destroyAllWindows()
    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)


