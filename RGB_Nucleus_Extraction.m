clear all;
close all;
clc;

mask_dir = 'Y:\HSI_Nucleus\SCC_Journal\Masks_Drawn\';
image_dir = 'Y:\HSI_Nucleus\SCC_Journal\PNGs\';
data_dir = 'Y:\HSI_Nucleus\SCC_Journal\Data\';

% Registration matrix
matrix = dlmread(strcat(image_dir,'Matrix.mat'));
aff = matrix([1,2,4],[1,2,4]);
aff(1,2)=-aff(1,2);
aff(2,1)=-aff(2,1);
tform = affine2d(aff);

% All annotated nucleus mask images
list = dir('Y:\HSI_Nucleus\SCC_Journal\Masks_Drawn\*.png');

for i = 1:length(list)
    
    filename = fullfile(list(i).folder,list(i).name);
    [folder,name,ext] = fileparts(filename);
    name_s = erase(name,'_syn');
    
    im1 = imread(strcat(image_dir,name,'.png'));
    im2 = imread(strcat(image_dir,name_s,'.png'));
    im2 = fliplr(im2);
    im2 = im2double(im2);
    
    mov_reg = imwarp(im2,tform);
    mov_reg_trans = imtranslate(mov_reg, [aff(3,1) aff(3,2)]);
    x = floor((size(mov_reg_trans,2)-size(im1,2))/2);
    y = floor((size(im1,1)-size(mov_reg_trans,1))/2);
    mov_reg_trans_2 = imtranslate(mov_reg_trans,[-x,15]);

    im2_reg = zeros(2000,2000,3);
    im2_reg(1:size(mov_reg_trans_2,1),:,:) = mov_reg_trans_2(:,1:2000,:);
    imshow(im2_reg)
    imshow(imfuse(im1,im2_reg))
    
    imwrite(im2_reg,strcat(image_dir,name_s,'_reg.png'));
    
    % Extract RGB nucleus patches
    im = imread(filename);
    mask = (im(:,:,2)==255)&(sum(im,3)==255);
    mask = imfill(mask,'holes');
    stats = regionprops(mask,'centroid');
    centroids = cat(1,stats.Centroid);% centroids: (x,y)
    centroids = floor(centroids/1);
    for k = 1:length(centroids)
        x = centroids(k,1);
        y = centroids(k,2);
        if (x>50)&&(x<1951)&&(y>50)&&(y<1951)
           patch = im2_reg(y-50:y+50,x-50:x+50,:); 
           imwrite(patch,strcat(data_dir,name_s,'_',num2str(k),'.png'));
        end
    end
    
    close all;
end

