import os
andrew_id = 'jiatong2'


def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False
    

if ( check_file('BRIEF.py') and \
     check_file('keypointDetect.py') and \
     check_file('panoramas.py') and \
     check_file('planarH.py') and \
     check_file('../results/q6_1.npy') and \
     check_file('../results/testPattern.npy') and \
     check_file('../jiatong2_hw2.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#you should also include files for extra credits if you are doing them (this check file does not check for them)
#images should be be included in the report
