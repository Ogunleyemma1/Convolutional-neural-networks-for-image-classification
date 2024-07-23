%Reference: https://uk.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
function Assignment6
%--- Load and Explore Image Data ---
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%--- Display a random sample of the data set ---
figure;
perm = randperm(10000,9);
for i = 1:9
    subplot(3,3,i);
    imshow(imds.Files{perm(i)});
    title(imds.Labels(perm(i)));
end

%table that contains the labels and the number of images having each label
labelCount = countEachLabel(imds) %show class labels & num. images per class

%Each image is 28-by-28-by-1 pixels
img = readimage(imds,1);
imageSize = size(img) %show image size

%--- Specify Training and Validation Sets ---
numTrainFiles = 750; %There are 1000 sample images per class in total
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%--- Define Network Architecture ---
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%--- Specify Training Options ---
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%--- Train Network Using Training Data ---
net = trainNetwork(imdsTrain,layers,options);

%--- Classify Validation Images and Compute Accuracy ---
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)



end