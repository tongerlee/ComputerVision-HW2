import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    A = []
    n = p1.shape[1]
    for i in range(n):
       x = p1[0, i]
       y = p1[1, i]
       u = p2[0, i]
       v = p2[1, i]
       A.append([-u, -v, -1, 0, 0, 0, u*x, v*x, x])
       A.append([ 0, 0, 0,-u, -v, -1, u*y, v*y, y])

    U, S, V = np.linalg.svd(np.asarray(A))
    H2to1 = V[-1, :]/V[-1, -1]

    return H2to1.reshape(3, 3)

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    bestH = None
    maxInliers = 0
    correctMatches = matches.shape[0]
    
    allX = locs1[matches[:, 0], 0:2]
    allU = locs2[matches[:, 1], 0:2]
    p1 = allX.swapaxes(0, 1)
    p2 = allU.swapaxes(0, 1)
    
    for i in range(num_iter):
        correspondences = np.random.randint(correctMatches, size=4)
        # print(correspondences)
        # print(p1[:, correspondences])
        # print(p2[:, correspondences])
        H2to1 = computeH(p1[:, correspondences], p2[:, correspondences])
        
        input_u = np.vstack((p2, np.ones(allU.shape[0])))
        test_x = np.matmul(H2to1, input_u)
        test_x = test_x/test_x[2, :]
        # print(test_x.shape)
        test_x = test_x[:2, :]
        # print(test_x.shape)
        distance = np.sqrt(np.sum((p1 - test_x)**2, axis = 0))
        
        currInliers = 0
        for each in distance:
            if each < tol:
                currInliers +=1
            #print(currInliers)
        
        if currInliers > maxInliers: 
            # print(H2to1)
            # print(currInliners)
            bestH, maxInliers = H2to1, currInliers

    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    print(bestH)

