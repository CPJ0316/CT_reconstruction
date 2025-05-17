import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
# https://chatgpt.com/share/681f89d1-a3f4-8006-9272-f400f93e72a6
# 讀取並預處理影像
img = imread("C:\\Users\\User\\Desktop\\toy.jpg")
img_gray = rgb2gray(img)
img_resized = resize(img_gray, (512, 512))

# 設定角度（theta）
theta_3 = np.arange(0., 180., 3)
theta_03 = np.arange(0., 180., 0.3)

# 建立 sinogram
sinogram_3 = radon(img_resized, theta=theta_3, circle=False)
sinogram_03 = radon(img_resized, theta=theta_03, circle=False)

# 重建（Ram-Lak 濾波）
nof_reconstruction_3 = iradon(sinogram_3, theta=theta_3, filter_name=None, output_size=512, circle=False)
nof_reconstruction_03 = iradon(sinogram_03, theta=theta_03, filter_name=None, output_size=512, circle=False)

# 重建（Ram-Lak 濾波）
reconstruction_3 = iradon(sinogram_3, theta=theta_3, filter_name='ramp', output_size=512, circle=False)
reconstruction_03 = iradon(sinogram_03, theta=theta_03, filter_name='ramp', output_size=512, circle=False)

# 顯示結果
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# 第一列：每 3 度
axs[0, 0].imshow(img_resized, cmap='gray')
axs[0, 0].set_title("Original Image (3°)")

axs[0, 1].imshow(sinogram_3, cmap='gray', aspect='auto')
axs[0, 1].set_title("Sinogram (3°)")

axs[0, 2].imshow(nof_reconstruction_3, cmap='gray')
axs[0, 2].set_title("Reconstructed Image (3°) no filter")

axs[0, 3].imshow(reconstruction_3, cmap='gray')
axs[0, 3].set_title("Reconstructed Image (3°)")

# 第二列：每 0.3 度
axs[1, 0].imshow(img_resized, cmap='gray')
axs[1, 0].set_title("Original Image (0.3°)")

axs[1, 1].imshow(sinogram_03, cmap='gray', aspect='auto')
axs[1, 1].set_title("Sinogram (0.3°)")

axs[1, 2].imshow(nof_reconstruction_03, cmap='gray')
axs[1, 2].set_title("Reconstructed Image (0.3°) no filter")

axs[1, 3].imshow(reconstruction_03, cmap='gray')
axs[1, 3].set_title("Reconstructed Image (0.3°)")

# 移除座標軸
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()