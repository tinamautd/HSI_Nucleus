close all;

wavelengths = [470, 472.8859, 475.7718, 478.6577, 481.5436, 484.4295, 487.3154, 490.2013, 493.0872, 495.9732, 498.8591, 501.745, 504.6309, 507.5168, 510.4027, 513.2886, 516.1745, 519.0604, 521.9463, 524.8322, 527.7181, 530.604, 533.4899, 536.3758, 539.2617, 542.1477, 545.0336, 547.9195, 550.8054, 553.6913, 556.5772, 559.4631, 562.349, 565.2349, 568.1208, 571.0067, 573.8926, 576.7785, 579.6644, 582.5503, 585.4362, 588.3221, 591.2081, 594.094, 596.9799, 599.8658, 602.7517, 605.6376, 608.5235, 611.4094, 614.2953, 617.1812, 620.0671, 622.953, 625.8389, 628.7248, 631.6107, 634.4966, 637.3826, 640.2685, 643.1544, 646.0403, 648.9262, 651.8121, 654.698, 657.5839, 660.4698, 663.3557, 666.2416, 669.1275, 672.0134, 674.8993, 677.7852, 680.6711, 683.557, 686.443, 689.3289, 692.2148, 695.1007, 697.9866, 700.8725, 703.7584, 706.6443, 709.5302, 712.4161, 715.302, 718.1879, 721.0738, 723.9597, 726.8456, 729.7315, 732.6174, 735.5034, 738.3893, 741.2752, 744.1611, 747.047, 749.9329, 752.8188, 755.7047, 758.5906, 761.4765, 764.3624, 767.2483, 770.1342, 773.0201, 775.906, 778.7919, 781.6779, 784.5638, 787.4497, 790.3356, 793.2215, 796.1074, 798.9933, 801.8792, 804.7651, 807.651, 810.5369, 813.4228, 816.3087, 819.1946, 822.0805, 824.9664, 827.8523, 830.7383, 833.6242, 836.5101, 839.396, 842.2819, 845.1678, 848.0537, 850.9396, 853.8255, 856.7114, 859.5973, 862.4832, 865.3691, 868.255, 871.1409, 874.0268, 876.9128, 879.7987, 882.6846, 885.5705, 888.4564, 891.3423, 894.2282, 897.1141, 900];
wavepick = wavelengths<721;
width_hsi=2000;
height_hsi=2000;
bands_hsi=150;%84;%
n = width_hsi * height_hsi * bands_hsi;

list_hsi = dir('F:\Raw_40x_new\**\*.raw');
list_mask= dir('Y:\HSI_Nucleus\SCC_Journal\Masks_Drawn\*.png');
% list_nuc = dir('Y:\HSI_Nucleus\SCC_Journal\Data\**\*.mat');

DATA_DIR = 'Y:\HSI_Nucleus\SCC_Journal\Data\';
DATA_SP_DIR = 'Y:\HSI_Nucleus\SCC_Journal\Data_SP\';

count = 0; % Count/Index of all nucleus patches 
last_pt_ind = '';

for i = 1:length(list_mask)
    
    % Mask image one-by-one
    filename_im = fullfile(list_mask(i).folder,list_mask(i).name);
    [folder,name,ext] = fileparts(filename_im);
    name_s = erase(name,'_syn');
    
    %-------- PT INDEX ----------
    pt_ind = name_s(1:5);
    if endsWith(pt_ind,'_')
        pt_ind = pt_ind(1:4);
    end
    
    if ~strcmp(pt_ind,last_pt_ind)
        if ~isempty(last_pt_ind)
           save(strcat(DATA_SP_DIR,last_pt_ind,'.mat'),'sp','lb','-v7.3');
           disp([last_pt_ind,' spectra saved!']);
           
           shadedErrorBar(...
               wavelengths(wavepick),...
               sp(lb==1,:),{@mean,@std},'lineprops','-r');
           hold on;
           shadedErrorBar(...
               wavelengths(wavepick),...
               sp(lb==0,:),{@mean,@std},'lineprops','-g');
           hold off;
           saveas(gcf,strcat(DATA_SP_DIR,last_pt_ind,'.png'));
           close(gcf);
           
        end
        count = 0;
        last_pt_ind = pt_ind;
        sp = single(zeros(0,87));
        lb = uint8(zeros(0,1));
    end
    
    %-----------------------------
    
    % Read mask image
    im = imread(filename_im);
    mask = (im(:,:,2)==255)&(sum(im,3)==255);
    mask = imfill(mask,'holes');
    
    % Locate HSI image
    hsifile = list_hsi(ismember({list_hsi.name},strcat(name_s,'.raw'))); 
    filename_hsi = fullfile(hsifile.folder,hsifile.name);
    [folder_hsi,name_hsi,ext_hsi] = fileparts(filename_hsi);
    
    % Read HSI image
    fid = fopen(filename_hsi,'r');
    f=fread(fid,n,'single');
    n_data = reshape(f,[height_hsi,width_hsi,bands_hsi]);
    if max(n_data(:))<1
        n_data = n_data*3;
    end
    n_data(n_data>1)=1;
    n_data(n_data<0)=0;
    n_data(isnan(n_data))=0;
    n_data(isinf(n_data))=1;
   
    % Extract nuclei
    stats = regionprops(mask,'centroid');
    centroids = cat(1,stats.Centroid);% centroids: (x,y)
    centroids = floor(centroids/1);
    
    switch folder_hsi(end)
        case 'N'
            label = 0;
        case 'T'
            label = 1;
    end
    
    for k = 1:length(centroids)
        
        nuc_id = strcat(DATA_DIR,pt_ind,'\',name_s,'_',num2str(k),'.mat');
        
        if isfile(nuc_id)
            
            x = centroids(k,1);
            y = centroids(k,2);
            
            if (x>50)&&(x<1951)&&(y>50)&&(y<1951)
                
                count = count + 1;
                
                patch = n_data(y-50:y+50,x-50:x+50,wavepick); 
                patch = single(patch);
                mask_patch = mask(y-50:y+50,x-50:x+50);
                temp(:) = sum(sum(BlackHole(patch,~mask_patch)))/sum(sum(mask_patch));
                sp(count,:) = temp;
                lb(count) = label;
            
                A = {strcat(name_s,'_',num2str(k)),label};
                xlRange = strcat('A',num2str(count));
                xlswrite(strcat(DATA_SP_DIR,'data.xlsx'),A,pt_ind,xlRange);
                
            end
            
        end
        
    end

end

% Save the data of the last PT
save(strcat(DATA_SP_DIR,pt_ind,'.mat'),'sp','lb','-v7.3');
disp([pt_ind,' spectra saved!']);

shadedErrorBar(...
   wavelengths(wavepick),...
   sp(lb==1,:),{@mean,@std},'lineprops','-r');
hold on;
shadedErrorBar(...
   wavelengths(wavepick),...
   sp(lb==0,:),{@mean,@std},'lineprops','-g');
hold off;
saveas(gcf,strcat(DATA_SP_DIR,pt_ind,'.png'));
close(gcf);