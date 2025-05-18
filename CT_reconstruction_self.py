import numpy as np
from numpy.fft import fft, ifft
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def radon_transform(image, theta):
    rows, cols = image.shape
    diagonal = int(np.sqrt(rows**2 + cols**2))
    pad_width = (diagonal - rows) // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0) #padding

    projections = [] #save projection result
    for angle in theta:
        rotated_image = rotate(padded_image, -angle, reshape=False, order=1)# rotate image
        projection = np.sum(rotated_image, axis=0) # get projection of one angle
        projections.append(projection) 
    return np.array(projections)

def inverse_radon_transform(sinogram, theta, size,filter):
    num_projections, num_detectors = sinogram.shape
    reconstruction = np.zeros((size, size))
    center = size // 2 # backprojection from the image center
    freqs = np.fft.fftfreq(num_detectors)
    ramp_filter = np.abs(freqs)

    for i in range(len(theta)):
        projection = sinogram[i, :]
        projection_fft = fft(projection) #Fourier transform
        if(filter==0):
            filtered_projection=np.real(projection)# no filter
        elif(filter==1):
            #filter in frequency domain
            filtered_projection = np.real(ifft(projection_fft * ramp_filter)) # multiply by filter、inverse Fourier transform
        else:
            #filter in spatial domain
            spatial_filter = np.real(np.fft.fftshift(ifft(ramp_filter)))#inverse Fourier transform、move DC to the center
            filtered_projection = np.convolve(projection, spatial_filter, mode='same') #spatial domain do convolution

        angle = np.deg2rad(theta[i]+90)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        #backprojection
        for x in range(size):
            for y in range(size):
                t = (x - center) * cos_a + (y - center) * sin_a #t can be + or -
                #num_detectors // 2=>let the detector's center corresponds to the middle of the index
                t_idx = int(round(t + num_detectors // 2)) 
                if 0 <= t_idx < num_detectors:
                    reconstruction[x, y] += filtered_projection[t_idx]
    return reconstruction


img = rgb2gray(imread("./image.jpg"))
img_resized = resize(img, (512, 512))
theta_03 = np.arange(0., 180., .3)
sinogram_03 = radon_transform(img_resized, theta_03)
n_reconstruction_03 = inverse_radon_transform(sinogram_03, theta_03, size=512,filter=0)
f_reconstruction_03 = inverse_radon_transform(sinogram_03, theta_03, size=512,filter=1)
s_reconstruction_03 = inverse_radon_transform(sinogram_03, theta_03, size=512,filter=2)

theta_3 = np.arange(0., 180.,3)
sinogram_3 = radon_transform(img_resized, theta_3)
n_reconstruction_3 = inverse_radon_transform(sinogram_3, theta_3, size=512,filter=0)
f_reconstruction_3 = inverse_radon_transform(sinogram_3, theta_3, size=512,filter=1)
s_reconstruction_3 = inverse_radon_transform(sinogram_3, theta_3, size=512,filter=2)
# 顯示結果
fig, axs = plt.subplots(2, 4, figsize=(15, 5))
axs[0,0].imshow(img_resized, cmap='gray')
axs[0,0].set_title('Original Image')
#axs[0,1].imshow(sinogram_3.T, cmap='gray', aspect='auto')
axs[0,1].imshow(n_reconstruction_3, cmap='gray')
axs[0,1].set_title('Reconstructed Image 3(without filter)')
axs[0,2].imshow(f_reconstruction_3, cmap='gray')
axs[0,2].set_title('Reconstructed Image 3_frequency domain')
axs[0,3].imshow(s_reconstruction_3, cmap='gray')
axs[0,3].set_title('Reconstructed Image 3_spatial domain')

axs[1,0].imshow(img_resized, cmap='gray')
axs[1,0].set_title('Original Image')
#axs[1,1].imshow(sinogram_03.T, cmap='gray', aspect='auto')
axs[1,1].imshow(n_reconstruction_03, cmap='gray')
axs[1,1].set_title('Reconstructed Image 0.3(without filter)')
axs[1,2].imshow(f_reconstruction_03, cmap='gray')
axs[1,2].set_title('Reconstructed Image 0.3_frequency domain')
axs[1,3].imshow(s_reconstruction_03, cmap='gray')
axs[1,3].set_title('Reconstructed Image 0.3_spatial domain')
plt.tight_layout()
plt.show()
