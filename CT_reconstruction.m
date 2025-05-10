
img=imread("D:\1成大\0課程\113-2\醫學影像系統\HW2\toy.jpg");
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
subplot(2, 3, 1);
% left image
imagesc(theta_3,xp_3,R_3);
xlabel('Angle (degrees)');
ylabel('Sensor Position');
title('Sinogram3');
colormap gray;
colorbar;
% right image
subplot(2, 3, 4);
imagesc(theta_03,xp_03,R_03);
xlabel('Angle (degrees)');
ylabel('Sensor Position');
title('Sinogram03');
colormap gray;
colorbar;

% back-projection
recon_unfiltered_3 = iradon(R_3, theta_3, 'linear', 'none');  % 'none' = no filter
recon_unfiltered_03 = iradon(R_03, theta_03, 'linear', 'none');

%figure;%new window
% left image
subplot(2, 3, 2);
imshow(recon_unfiltered_3, []);
title('Sinogram3');
% right image
subplot(2, 3, 5);
imshow(recon_unfiltered_03, []);
title('Sinogram03');

% back-projection
recon_filtered_3 = iradon(R_3, theta_3, 'linear', 'Ram-Lak');  % 'none' = no filter
recon_filtered_03 = iradon(R_03, theta_03, 'linear', 'Ram-Lak');

% left image
subplot(2, 3, 3);
imshow(recon_filtered_3, []);
title('Sinogram3 filtered');
% right image
subplot(2, 3, 6);
imshow(recon_filtered_03, []);
title('Sinogram03 filtered');