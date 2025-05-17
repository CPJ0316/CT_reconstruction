import numpy as np
from numpy.fft import fft, ifft
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

'''
def radon_transform(image, theta):
    num_detectors = image.shape[0]
    sinogram = np.zeros((len(theta), num_detectors))
    for i, angle in enumerate(theta):
        rotated = rotate(image, -angle, reshape=False, order=1)
        sinogram[i, :] = np.sum(rotated, axis=0)
    return sinogram
'''
def radon_transform(image, theta):
    rows, cols = image.shape
    diagonal = int(np.sqrt(rows**2 + cols**2))
    pad_width = (diagonal - rows) // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    projections = []
    for angle in theta:
        rotated_image = rotate(padded_image, -angle, reshape=False, order=1)
        projection = np.sum(rotated_image, axis=0)
        projections.append(projection)
    return np.array(projections)

def inverse_radon_transform(sinogram, theta, size,filter):
    num_projections, num_detectors = sinogram.shape
    reconstruction = np.zeros((size, size))
    center = size // 2
    freqs = np.fft.fftfreq(num_detectors)
    ramp_filter = np.abs(freqs)

    for i in range(len(theta)):#num_projections
        projection = sinogram[i, :]
        projection_fft = fft(projection)
        if(filter):
            filtered_projection = np.real(ifft(projection_fft * ramp_filter))
        else:
            filtered_projection = np.real(ifft(projection_fft))  # 去掉滤波器
        angle = np.deg2rad(theta[i]+90)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for x in range(size):
            for y in range(size):
                t = (x - center) * cos_a + (y - center) * sin_a
                t_idx = int(round(t + num_detectors // 2))
                if 0 <= t_idx < num_detectors:
                    reconstruction[x, y] += filtered_projection[t_idx]

    reconstruction *= np.pi / (2 * len(theta))
    return reconstruction

# 主程式
img = rgb2gray(imread("C:\\Users\\User\\Desktop\\toy.jpg"))
img_resized = resize(img, (512, 512))
theta = np.arange(0., 180., .3)
sinogram = radon_transform(img_resized, theta)
nof_reconstruction = inverse_radon_transform(sinogram, theta, size=512,filter=False)
reconstruction = inverse_radon_transform(sinogram, theta, size=512,filter=True)
# 顯示結果
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(img_resized, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(sinogram.T, cmap='gray', aspect='auto')
axs[1].set_title('Sinogram')
axs[2].imshow(nof_reconstruction, cmap='gray')
axs[2].set_title('Reconstructed Image(no filter)')
axs[3].imshow(reconstruction, cmap='gray')
axs[3].set_title('Reconstructed Image')
plt.tight_layout()
plt.show()
