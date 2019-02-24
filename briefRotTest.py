import numpy as np
import cv2
import BRIEF

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    compareX, compareY = BRIEF.makeTestPattern()
    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = BRIEF.briefLite(im)  
    
    dst = cv2.imread('../data/model_chickenbroth.jpg')
    rotMatches = []
    rows = dst.shape[0]
    cols = dst.shape[1]
    locsd, descd = BRIEF.briefLite(dst)
    matches = BRIEF.briefMatch(desc, descd)
    rotMatches.append(matches.shape[0])
    fig = plt.figure(0)
    tmp = fig.add_subplot(6, 6, 1)
    plt.imshow(dst)
    for i in range(35):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 10* (i+1), 1)
        dst_new = cv2.warpAffine(dst, M, (cols, rows))
        locsd, descd = BRIEF.briefLite(dst_new)
        matches = BRIEF.briefMatch(desc, descd)
        rotMatches.append(matches.shape[0])
        tmp = fig.add_subplot(6, 6, i+2)
        plt.imshow(dst_new)
    # fig.savefig('../results/rotation.png')
    plt.show()
    
    x_label = ['0', '10', '20', '30', '40', \
            '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', \
            '150', '160', '170', '180', '190', '200', '210', '220', '230', \
            '240', '250', '260', '270', '280', '290', '300', '310', '320', \
            '330', '340', '350']
    index = np.arange(len(x_label))
    plt.figure(1)
    plt.bar(index, rotMatches, align = 'center', alpha = 0.5)
    plt.xlabel('Degree')
    plt.ylabel('Number of correct matches')
    plt.xticks(index, x_label, fontsize = 5, rotation = 30)
    plt.title('Rotation angle vs the number of correct matches')
    plt.show()