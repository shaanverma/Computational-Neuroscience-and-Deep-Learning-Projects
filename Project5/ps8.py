"""
AM4264 - Problem Set 8
Name: Shaan Verma
Student #: 250804514
Date: March 15, 2021
"""

import matplotlib.pyplot as plt
from scipy.io import loadmat
import hmax
import pickle
import numpy as np

'''
#############
#           #
# Problem 1 #
#           #
#############

# load and examine the HMAX output
output = pickle.load( open('output.pkl', 'rb'))

s1 = output['s1']
n_sscales = len(s1) # s = spatial
s1_sscale1 = s1[0]
s1_sscale1.shape

# x y z vales in the diagram
x = np.shape(s1_sscale1)[2]
y = np.shape(s1_sscale1)[3]
z = np.shape(s1_sscale1)[1]

c1 = output['c1']

# m value
#n =np.shape(c1[6])[3])
m = np.shape(c1[6])[2]

print(np.shape(c1[7]))

# n value
#n =np.shape(c1[7])[3])c1 = output['c1']
n = np.shape(c1[7])[2]


s2 = output['s2']
n_fscales = len(s2) # f = filter
s2_fscale1 = s2[7]
n_sscales = len(s2_fscale1)
s2_fscale1_sscale1 = s2_fscale1[7]
s2_fscale1_sscale1.shape

# s value
s = s2_fscale1_sscale1.shape[1]

operationA = "Tuning operation (Gaussian)"
operationB = "local max pooling"

# save answers
# insert x, y, z, m, n, s (integers), and layer name, operation A, operation B (strings) ...
#    ... as values in the dictionary below and save the dictionary as a text file using the code below
dict = {'x':x  , 'y':y  , 'z':z  , 'm':m  , 'n':n  , 's':s  , 'layer name':'C1'  , 'operation A':operationA  , 'operation B': operationB  } 
f = open('ps8_prob1.txt', 'w')
f.writelines( [str(dict['x'])+'\n', str(dict['y'])+'\n', str(dict['z'])+'\n', str(dict['m'])+'\n', str(dict['n'])+'\n', str(dict['s'])+'\n', dict['layer name']+'\n', dict['operation A']+'\n', dict['operation B']] )
f.close()


# content of solution_code_ps8.py
def prob1_answers(i):
    f = open('ps8_prob1.txt', 'r')
    text = f.readlines()
    f.close() 
    
    return text[i]

'''




'''

##############################
#                            #
# Problem 2: Layer Responses #
#                            #
##############################

# Loading output pickle file from problem 1
output = pickle.load( open('output.pkl', 'rb'))


#S1 Response 
#SpatialScales = 7 and 8
#FilterType = 2

s1 = output['s1']
spScale7 = s1[6]
spScale8 = s1[7]

s1_image1 = spScale7[1][1][:][:]
s1_image2 = spScale8[1][1][:][:]

plt.figure(); plt.imshow(s1_image1); plt.title('S1 Response - FilterType=2, SpatialScale=7')
plt.figure(); plt.imshow(s1_image2); plt.title('S1 Response - FilterType=2, SpatialScale=8')


#C1 Response 
#SpatialScale = 4
#FilterType = 2

c1 = output['c1']
c1SpScale4 = c1[3]
c1_image = c1SpScale4[1][1][:][:]
plt.figure(); plt.imshow(c1_image); plt.title('C1 Response - FilterType=2, SpatialScale=4')


#S2 Response 
#SpatialScale = 4
#FilterScale = 4
#FilterType = 111

s2 = output['s2']
s2SpFilterScale = s2[3][3]
s2_image = s2SpFilterScale[1][110][:][:]
plt.figure(); plt.imshow(s2_image); plt.title('S2 Response - FType=111, FScale=4 ,SP_Scale=4')


##############################
#                            #
#     Problem 2: Filters     #
#                            #
##############################


###############
## S1 Filter ##
###############
# Calling garbor_filter function from hmax.py file
#HMAX = hmax.gabor_filter(250,3.7,90)

filters1 = hmax.S1(size=19, wavelength=3.7) # spatial scale 7
filters2 = hmax.S1(size=21, wavelength=3.65) # spatial scale 8

# showing 1st filter
filter1 = filters1.gabor.weight.data[1,0,:,:]  
plt.figure(); plt.imshow(filter1, cmap=plt.gray()); plt.title('S1 Filter - Size=19, Wavelength=3.7 (SPscale 7)')

# showing second filter
filter2 = filters2.gabor.weight.data[1,0,:,:]  
plt.figure(); plt.imshow(filter2, cmap=plt.gray()); plt.title('S1 Filter - Size=21, Wavelength=3.65 (SPscale 8)')

###############
## S2 Filter ##
###############
# load patches (= filters)
m = loadmat('universal_patch_set.mat') 
patches = [patch.reshape(shape[[2, 1, 0, 3]]).transpose(3, 0, 2, 1)
           for patch, shape in zip(m['patches'][0], m['patchSizes'].T)]
n_fscales = len(patches)
patches_fscale1 = patches[3]
patches_fscale1.shape

# show one of the 400 filters (filters are 3D instead of 2D)
patch = patches_fscale1[110,:,:,:]
fig, axs = plt.subplots(1, 4)
for dim3i in range(0,4):
    axs[dim3i].imshow(patch[dim3i,:,:], cmap=plt.gray())
    axs[dim3i].set_xlabel('component ' + str(dim3i+1))
    if dim3i == 0:
        axs[dim3i].set_title('S2 filter - 111th Filter')

###############
## C1 Filter ##
###############


'''

