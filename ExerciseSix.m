function ExerciseSix
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
    
    % Table that contains the labels and the number of images having each label
    labelCount = countEachLabel(imds); % show class labels & num. images per class
    
    % Each image is 28-by-28-by-1 pixels
    img = readimage(imds,1);
    imageSize = size(img); % show image size
    
    %--- Specify Training and Validation Sets ---
    numTrainFiles = 750; % There are 1000 sample images per class in total
    [imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');
    
    %--- Define Enhanced Network Architecture ---
    layers = [
        imageInputLayer([28 28 1])
        
        % First block
        convolution2dLayer(3, 16, 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        % Second block
        convolution2dLayer(3, 32, 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        % Third block
        convolution2dLayer(3, 64, 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        % Fourth block
        convolution2dLayer(3, 128, 'Padding', 'same')
        reluLayer
        
        % Flatten and Fully Connected Layers
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(10)  % Number of classes
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
    net = trainNetwork(imdsTrain, layers, options);
    
    %--- Classify Validation Images and Compute Accuracy ---
    YPred = classify(net, imdsValidation);
    YValidation = imdsValidation.Labels;
    
    accuracy = sum(YPred == YValidation) / numel(YValidation);
    fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);
    
    %--- Classify Custom Test Images ---
    customImagesPath = 'C:\Users\pc\Documents\DIGITAL ENGINEERING\Summer 2024\Image Analysis and Processing\Exercise\Exercise6\Mytest'; % Set this path to your custom images folder
    customImds = imageDatastore(customImagesPath, ...
        'IncludeSubfolders',true,'LabelSource','none');
    
    % Resize custom images to 28x28 if they are not already
    numFiles = numel(customImds.Files);
    customImages = zeros(28, 28, 1, numFiles);
    for i = 1:numFiles
        img = readimage(customImds, i);
        if size(img, 1) ~= 28 || size(img, 2) ~= 28
            img = imresize(img, [28 28]);
        end
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        customImages(:, :, 1, i) = img;
    end
    
    % Convert to datastore
    customImdsResized = augmentedImageDatastore([28 28 1], customImages);

    % Classify custom images
    customPredictions = classify(net, customImdsResized);

    % Display custom images and their predicted labels
    figure;
    for i = 1:numFiles
        subplot(ceil(numFiles / 3), 3, i);
        imshow(customImages(:, :, 1, i));
        title(string(customPredictions(i)));
    end
end
