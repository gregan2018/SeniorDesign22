function TBDetectionTest

rng(0);

%% Read all data from single folder

folder = 'C:\Users\admin\Documents\MATLAB\SeniorDesign22\TrainVal\NoPreProc\Test\';

tb = dir([folder, 'tb\', '*', '.png']);
hy = dir([folder  'health\', '*', '.png']);
data = [tb;hy];
imds = imageDatastore(fullfile({data.folder}.', {data.name}.'));
lbls = [ones(numel(tb),1); zeros(numel(hy),1)];
imds.Labels = categorical(lbls, [0,1], {'health','tb'});

% Number of Images
numImages=length(imds.Labels);

% Visualize random images
perm=randperm(numImages,6);

figure;

for idx=1:length(perm)
    
    subplot(2,3,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end

%% 

% Loading Saved Model
load(['TBDetectNoPreProc.mat'],'net');

% Preprocessing the images
imds.ReadFcn = @(filename)Sub_PreprocessXray(filename);

predictions = predict(net, imds);

% perfcurve(lbls, predictions);
accuracy = mean(lbls == predictions(:,2))>0.5;

disp(['Accuracy: ' num2str(accuracy)]);
end

%%
function Iout = Sub_PreprocessXray(filename)
% This function preprocesses the given X-ray image by converting it into
% grayscale if required and later converting to 3-channel image to adapt to
% existing deep learning architectures
%
% This function will likely change depending upon the pretrained network used
%
% Also resizes so that the images match the input that is expected by the network


% Read the Filename
I = imread(filename);

% Some images might be RGB, convert them to Grayscale
if ~ismatrix(I)
    I=rgb2gray(I);
end

I = single(I)/255;

I = imresize(I,[224 224]);
% Replicate the image 3 times to create an RGB image
Iout = cat(3,I,I,I);

end