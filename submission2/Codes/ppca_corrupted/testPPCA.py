import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv
import numpy as np
import PPCA
import skimage
from skimage import color
from sklearn.metrics import mean_squared_error
from math import sqrt



RGB = mpimg.imread("CameraMan.png")
print RGB.shape
#print(RGB)
#plt.imshow(RGB)
#plt.show()
#RGB_noise = mpimg.imnoise(RGB,'salt & pepper',0.02)

#print(img)
RGB_noise = skimage.util.random_noise(RGB, mode='s&p', seed=None, clip=True, amount=0.50)
RGB_Gray = color.rgb2gray(RGB_noise)
#RGB_double = RGB_Gray.astype(dtype = np.double)
#print(RGB_double)
#print('double done')

# Required Number of Principal axis.
k = 20

# Applying PPCA with EM on data corrupted data matrix.
[W,sigma_square,Xn,t_mean,M] = PPCA.em_PPCA(RGB_Gray,k)
#print(M)
# Recovered Image
rec_Image = np.dot(np.dot(np.dot(W,inv(np.dot(W.T,W))),M),Xn)
rec_Image = rec_Image + np.dot((t_mean),np.ones((1,rec_Image.shape[1])))

#rec_Imagefinal = rec_Image.astype(dtype = np.uint8)
plt.imshow(rec_Image, cmap = plt.cm.Greys_r)
plt.show()

rms = sqrt(mean_squared_error(RGB, rec_Image))

print rms