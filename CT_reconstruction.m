
img=imread("./image.jpg");
img_resized=imresize(img,[512,512]);
if size(img_resized,3)==3
    img_gray=rgb2gray(img_resized);
else
    img_gray=img_resized;
end
%imshow(img_gray);
% define the angle
theta_3 =0:3:179;
theta_03=0:0.3:179;
%calculate sinogram
[R_3,xp_3]=radon(img_gray,theta_3);
[R_03,xp_03]=radon(img_gray,theta_03);

%show sinogram
figure;%new window
subplot(2, 4, 1);
% left image
imshow(img_gray, []);
title('Original Image');
% right image
subplot(2, 4, 5);
imshow(img_gray, []);
title('Original Image');


subplot(2, 4, 2);
% left image
imagesc(theta_3,xp_3,R_3);
xlabel('Angle (degrees)');
ylabel('Sensor Position');
title('Sinogram 3');
colormap gray;
colorbar;
% right image
subplot(2, 4, 6);
imagesc(theta_03,xp_03,R_03);
xlabel('Angle (degrees)');
ylabel('Sensor Position');
title('Sinogram 0.3');
colormap gray;
colorbar;

% back-projection, no filter
recon_unfiltered_3 = iradon(R_3, theta_3, 'linear', 'none');  % 'none' = no filter
recon_unfiltered_03 = iradon(R_03, theta_03, 'linear', 'none');

% left image
subplot(2, 4, 3);
imshow(recon_unfiltered_3, []);
title('Sinogram 3');
% right image
subplot(2, 4, 7);
imshow(recon_unfiltered_03, []);
title('Sinogram 0.3');

% back-projection
recon_filtered_3 = iradon(R_3, theta_3, 'linear', 'Ram-Lak');  
recon_filtered_03 = iradon(R_03, theta_03, 'linear', 'Ram-Lak');

% left image
subplot(2, 4, 4);
imshow(recon_filtered_3, []);
title('Sinogram 3 filtered');
% right image
subplot(2, 4, 8);
imshow(recon_filtered_03, []);
title('Sinogram 0.3 filtered');