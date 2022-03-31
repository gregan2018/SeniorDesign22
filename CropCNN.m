function net = CropCNN(trainImgFilePath, procImgFilePath, inputSz)

% Parameters
lr = 0.0001;
numEpochs = 50;

% Create image datastores
trainImgs = imageDatastore(trainImgFilePath, 'FileExtensions','.png', 'IncludeSubfolders',1);
msks = pixelLabelDatastore(procImgFilePath, ["true", "false"], [1, 0], 'IncludeSubfolders',1);
comb = combine(trainImgs, msks);

% Create encoder block
encoderBlock = @(block) [
    convolution2dLayer(3, 2^(2+block), "Padding", 'same')
    reluLayer
    convolution2dLayer(3, 2^(2+block), "Padding", 'same')
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)];
encoder = blockedNetwork(encoderBlock, 4, "NamePrefix", "encoder_");

% Create decoder block
decoderBlock = @(block) [
    transposedConv2dLayer(2, 2^(7-block), 'Stride', 2)
    convolution2dLayer(3, 2^(7-block), "Padding", 'same')
    reluLayer
    convolution2dLayer(3, 2^(7-block), "Padding", 'same')
    reluLayer];
decoder = blockedNetwork(decoderBlock, 4, "NamePrefix", "decoder_");

% Create bridge
bridge = [
    convolution2dLayer(3, 128, "Padding", 'same')
    reluLayer
    convolution2dLayer(3, 128, "Padding", 'same')
    reluLayer
    dropoutLayer(0.5)];    

% Create output layer
finalSoftMax = softmaxLayer('Name', 'FinalSoftMaxLayer');
finalClassificationLayer = pixelClassificationLayer('Name', 'FinalClassificationLayer');

% Create the unet architecture
unet = encoderDecoderNetwork(inputSz, encoder, decoder, ...
    "OutputChannels", 1, ...
    "SkipConnections", "concatenate", ...
    "LatentNetwork", bridge);

% Connect output layers
lgraph = layerGraph(unet);
lgraph = addLayers(lgraph, finalSoftMax);
lgraph = addLayers(lgraph, finalClassificationLayer);
lgraph = connectLayers(lgraph,'encoderDecoderFinalConvLayer', 'FinalSoftMaxLayer');
lgraph = connectLayers(lgraph,'FinalSoftMaxLayer', 'FinalClassificationLayer');

% Training options
trainOpts = trainingOptions('adam', ...
    'InitialLearnRate', lr, ...
    'MiniBatchSize', 1, ...
    'MaxEpochs', numEpochs, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(comb, lgraph, trainOpts);
end
