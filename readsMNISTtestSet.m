%----------------------------------------------------
% Prof. Saraiva, 2019/1
%----------------------------------------------------
% TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
%----------------------------------------------------
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000803(2051) magic number 
% 0004     32 bit integer  10000            number of images 
% 0008     32 bit integer  28               number of rows 
% 0012     32 bit integer  28               number of columns 
% 0016     unsigned byte   ??               pixel 
% 0017     unsigned byte   ??               pixel 
% ........ 
% xxxx     unsigned byte   ??               pixel
%----------------------------------------------------
% Pixels are organized row-wise. Pixel values are 0 to 255. 
% 0 means background (white), 255 means foreground (black). 
%----------------------------------------------------
%----------------------------------------------------
% TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
%----------------------------------------------------
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
% 0004     32 bit integer  10000            number of items 
% 0008     unsigned byte   ??               label 
% 0009     unsigned byte   ??               label 
% ........ 
% xxxx     unsigned byte   ??               label
%----------------------------------------------------
% The labels values are 0 to 9.
%----------------------------------------------------
clc; clear all; close all;

% opens binary file of test data
fid1 = fopen('..\..\DataSets\MNIST\t10k-images.idx3-ubyte');

% opens binary file of test data labels
fid2 = fopen('..\..\DataSets\MNIST\t10k-labels.idx1-ubyte');

% reads first four data fields in the test data file
header=fread(fid1,4,'int','b'); % big-endian (MSB)
N = header(2); % number of images
m = header(3); % number of lines
n = header(4); % number of columns

% reads first two data fields in the labels file
header=fread(fid2,2,'int','b'); % big-endian (MSB)
M = header(2); % number of labels

% initializes matrix that will store the images
TEST = zeros(m,n,N);

% initializes matrix that will store the labels
TESTLABELS = zeros(1,N);

% reads images and labels
for i=1:N
    TEST(:,:,i)=fread(fid1,[m n],'uint8');
    TESTLABELS(1,i)=fread(fid2,1,'uint8');
end;

% display a sample (first 100 images)
for i=1:100
    subplot(10,10,i);
    imshow(imcomplement(TEST(:,:,i)'));
end;    

% closes files
fclose(fid1);
fclose(fid2);

% saves test set and its labels
save MNIST_testSet TEST TESTLABELS;

