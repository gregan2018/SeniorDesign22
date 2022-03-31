function TBDetectionTrain
%% Loading images to a datastore to read images to memory only when required

imdsTrain = imageDatastore('C:\Users\admin\Documents\MATLAB\SeniorDesign22\TrainVal\NoPreProc\Train\', "IncludeSubfolders", true, "LabelSource", "folderNames");
imdsVal = imageDatastore('C:\Users\admin\Documents\MATLAB\SeniorDesign22\TrainVal\NoPreProc\Val\', "IncludeSubfolders", true, "LabelSource", "folderNames");

%%
% ResNet Architecture
load('resnet50.mat', 'net'); %net = resnet50;
lgraph = layerGraph(net);
clear net;

% Number of categories
numClasses = numel(categories(imdsTrain.Labels));

%% -- Changing the last layer(s) to use it for the new task
% 
% New Learnable Layer
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

% Replacing the last layers with new layers
lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);
newsoftmaxLayer = softmaxLayer('Name','new_softmax');
lgraph = replaceLayer(lgraph,'fc1000_softmax',newsoftmaxLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);

%% --

% Preprocessing the images
imdsTrain.ReadFcn = @(filename)Sub_PreprocessXray(filename);
imdsVal.ReadFcn = @(filename)Sub_PreprocessXray(filename);

% Training Options, we choose a small mini-batch size due to limited images  
% Look further into some of the training options, including CheckpointFile
options = trainingOptions('adam',...
    'MaxEpochs',30,'MiniBatchSize',16,...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ValidationData',imdsVal, ...
    'ValidationFrequency',30,...
    'ExecutionEnvironment','gpu');

% Data Augumentation    
% Look further into this
augmenter = imageDataAugmenter( ...
    'RandRotation',[-5 5],'RandXReflection',1,...
    'RandYReflection',1,'RandXShear',[-5 5],'RandYShear',[-5 5]);

% Resizing all training images to [224 224] for ResNet architecture
augImds = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter);

% Training
net = trainNetwork(augImds,lgraph,options);

% Saving Model
save('TBDetectNoPreProc.mat','net');

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

I = imresize(I,[224 224]);

% Divide by 255 to scale to 0 to 1;
I = single(I)/255;

% Replicate the image 3 times to create an RGB image
Iout = cat(3,I,I,I);
end