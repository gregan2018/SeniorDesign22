function net = TBCNN(trainImgFilePath, inputSz)

classes = ["B", "C", "L", "E", "H", "M", "N"];
classWgts = [1, 2, 2, 2, 2, 2, 2];
classWgts = classWgts/sum(classWgts);

% Create image datastores
trainImgs = imageDatastore([trainImgFilePath, 'Imgs/'], 'FileExtensions','.png');
trainMsks = pixelLabelDatastore([trainImgFilePath, 'Msks/'], ["B", "C", "L", "E", "H", "M", "N"], [0,1,2,3,4,5,6]);
trainComb = combine(trainImgs, trainMsks);

valImgs = imageDatastore([valImgFilePath, 'Images\'], 'FileExtensions','.png','IncludeSubfolders',1);
valMsks = pixelLabelDatastore([valImgFilePath, 'Masks\'], ["C", "L", "E", "H", "M", "N", "B"], [6,5,4,3,2,1,0]);
valComb = combine(valImgs, valMsks);

% Augment training data
augTrainImds = transform(trainComb, @augmentTrainImages);
augTrainImds = transform(augTrainImds, @centerCropImageAndLabel);
augValImds = transform(valComb, @centerCropImageAndLabel);

% Create encoder block
encoderBlock = @(block) [
    convolution2dLayer(3, 2^(3+block), "Padding", 'same', 'WeightsInitializer', 'he')
    reluLayer
    convolution2dLayer(3, 2^(3+block), "Padding", 'same', 'WeightsInitializer', 'he')
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)];
encoder = blockedNetwork(encoderBlock, 4, "NamePrefix", "encoder_");

% Create decoder block
decoderBlock = @(block) [
    resize2dLayer("Scale", 2)
    convolution2dLayer(3, 2^(8-block), "Padding", 'same', 'WeightsInitializer', 'he')
    reluLayer
    convolution2dLayer(3, 2^(8-block), "Padding", 'same', 'WeightsInitializer', 'he')
    reluLayer];
decoder = blockedNetwork(decoderBlock, 4, "NamePrefix", "decoder_");

% Create bridge
bridge = [
    convolution2dLayer(3, 256, "Padding", 'same')
    reluLayer
    convolution2dLayer(3, 256, "Padding", 'same')
    reluLayer
    dropoutLayer(0.5)];    

% Create output layer
finalSoftMax = softmaxLayer('Name', 'FinalSoftMaxLayer');
finalClassificationLayer = pixelClassificationLayer('Name', 'FinalClassificationLayer', 'Classes', classes, 'ClassWeights', classWgts);

% Create the unet architecture
unet = encoderDecoderNetwork(inputSz, encoder, decoder, ...
    "OutputChannels", 7, ...
    "SkipConnections", "concatenate", ...
    "LatentNetwork", bridge);

% Connect output layers
lgraph = layerGraph(unet);
lgraph = addLayers(lgraph, finalSoftMax);
lgraph = addLayers(lgraph, finalClassificationLayer);
lgraph = connectLayers(lgraph, 'encoderDecoderFinalConvLayer', 'FinalSoftMaxLayer');
lgraph = connectLayers(lgraph, 'FinalSoftMaxLayer', 'FinalClassificationLayer');

% Training options
options = trainingOptions('adam',...
    'MaxEpochs',200,'MiniBatchSize',32,...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-3, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ValidationData',augValImds);

net = trainNetwork(augTrainImds, lgraph, options);
end

function out = augmentTrainImages(data)
% Unpack original data.
I = data{1};
C = data{2};

% Define random affine transform.
tform = randomAffine2d("XReflection",true,"YReflection", true, 'Rotation',[-30, 30], 'XTranslation', [-10, 10], 'YTranslation', [-10, 10]);
rout = affineOutputView(size(I),tform);

% Transform image and bounding box labels.
augmentedImage = imwarp(I,tform,"OutputView",rout);
augmentedLabel = imwarp(C,tform,"OutputView",rout);

% Return augmented data.
out = {augmentedImage,augmentedLabel};
end

function out = centerCropImageAndLabel(data)
targetSize = [64, 64];
I = data{1};
C = data{2};
win = centerCropWindow2d(size(I),targetSize);
I = imcrop(I,win);
I = single(I)/255;
C = imcrop(C,win);
out = {I, C};
end
