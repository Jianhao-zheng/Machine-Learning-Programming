function [cimg, ApList, muList] = compress_image(img, p)
%COMPRESS_IMAGE Compress the image by applying the PCA over each channels 
% independently
%
%   input -----------------------------------------------------------------
%   
%       o img : (width x height x 3), an image of size width x height over RGB channels
%       o p : The number of components to keep during projection 
%
%   output ----------------------------------------------------------------
%
%       o cimg : (p x height x 3) The projection of the image on the eigenvectors
%       o ApList : (p x width x 3) The projection matrices for each channels
%       o muList : (width x 3) The mean vector for each channels
width=size(img,1);
height=size(img,2);
channel=size(img,3);
cimg=zeros(p,height,channel);
ApList=zeros(p,width,channel);
muList=zeros(width,channel);
for i_channel=1:channel
    [muList(:,i_channel), ~, EigenVectors, ~] = compute_pca(img(:,:,i_channel));
    [cimg(:,:,i_channel), ApList(:,:,i_channel)] = project_pca(img(:,:,i_channel), muList(:,i_channel), EigenVectors, p);
end

end

