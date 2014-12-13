clear all;
load('data.mat'); % load filter bank
load('dictionary.mat'); % load dictionary

%--------------------------------------
%   YOUR CODE STARTS HERE
%--------------------------------------

%path for your selected image
img_path = 'myBasilica.jpg';

%read the image
I = imread(img_path);

%extract feature points
featurePoints = extractFilterResponses(I, filterBank);

%TODO: find the nearest clusters for each feature vector
distances = pdist2(featurePoints,dictionary);
[~,closestCluster]=min(distances,[],2);
%TODO: build an "image of cluster belonging" (the segments)
imageSegments=reshape(closestCluster,[size(I,1), size(I,2)]);
%TODO: visualize and save the segments image

imagesc(imageSegments);
imwrite(imageSegments, jet(300), 'myBasilicaSegmented.jpg');

%--------------------------------------
%   YOUR CODE ENDS HERE
%--------------------------------------

