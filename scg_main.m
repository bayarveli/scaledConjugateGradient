%% Scaled Congugate Gradient Algorithm Implementation
clc
clear

format long
%% DATASET DEFINITION
load('SamsungIMUAllData.mat');

% mix dataset (optional)
allDataset = scg_mixDataset(SamsungIMUAllData, 320);

% divide dataset to train and test sets with 70% and 30%
[trainingSet, testSet] = scg_divideDataset(allDataset, 70, 30);

% training set
trainingSetInput = trainingSet(:,1:18);
% training set target classes
trainingSetTarget = trainingSet(:,19:27);
% test set
testSetInput = testSet(:,1:18);
% test set target classes
testSetTarget = testSet(:,19:27);

[trRowCount trColCount] = size(trainingSetInput);
%% INITIALIZE NETWORK

% WARNING! : Network settings is valid for 3-layer networks which are
% fixed as 1 Input Layer, 1 Hidden Layer and 1 Output Layer. In each layer,
% the number of nodes can be changed according to problem.
networkSettings.inputNodeCount = 18;
networkSettings.hiddenNodeCount = 32;
networkSettings.outputNodeCount = 9;
networkSettings.weightUpperLimit = 0.1;
networkSettings.weightLowerLimit = -0.1;
networkSettings.initialBiasValue = 0.0;
networkSettings.maxIteration = 1000;
networkSettings.errorLimit = 1.e-2;
networkSettings.sampleCount = trRowCount; % sample count in training set
networkSettings.preprocess = 1; % 1 : Standardize, 2 : Normalize

% Neural network is initialized with given settings
network = scg_initializeNetwork(networkSettings);

%% PREPROCESS
% get pre-processed training data set
if(network.preprocess == 1)
    trainingSetInput = scg_standardize(trainingSetInput);
elseif(network.preprocess == 2)
    trainingSetInput = scg_normalize(trainingSetInput);
end

%% TRAINING
% Scaled Conjugate Gradient (SCG) Algorithm is used as training algorithm.

% All weights in the network rearranged as weight vector.
weightVector = scg_getWeightVector(network);

% Calculate initial gradient value
gradient_new = -1 * scg_calculateGradient(network, weightVector, trainingSetInput, trainingSetTarget);

% Set algorithm initial parameters
sigma0 = 1.e-6;
lambda = 1.e-6;
lambda_bar = 0;
r_new = gradient_new;

% Flag for algorithm stopping criteria
success = true;
% iteration index
iterIdx = 0;

% SCG main loop #start#
while iterIdx < network.maxIteration
    iterIdx = iterIdx + 1;
    r = r_new;
    gradient = gradient_new;
    mu = dot(gradient,gradient);
    
    if(success)
        success = false;
        sigma = sigma0 / sqrt(mu);
        s = (scg_calculateGradient(network, (weightVector + sigma * gradient),trainingSetInput, trainingSetTarget) ...
            - scg_calculateGradient(network, weightVector, trainingSetInput, trainingSetTarget))/sigma;
        delta = dot(gradient', s);
    end
    
    % scale s
    zetta = lambda - lambda_bar;
    s = s + zetta * gradient;
    delta = delta + zetta * mu;
    
    if delta < 0
        s = s + (lambda - 2 * delta / mu) * gradient;
        lambda_bar = 2 * (lambda - delta / mu);
        delta = delta - lambda * mu;
        delta = delta * -1;
        lambda = lambda_bar;
    end
    
    phi = dot(gradient', r);
    alpha = phi / delta;
    
    vector_new = weightVector + alpha * gradient;
    
    % calculate error with weight vector
    f_old = scg_calculateError(network, weightVector, trainingSetInput, trainingSetTarget);
    % calculate error with updated weight vector
    f_new = scg_calculateError(network, vector_new, trainingSetInput, trainingSetTarget);
    % calculate comparison value
    comparison = 2 * delta * (f_old - f_new) / (phi ^ 2);
    
    if comparison >= 0
        if f_new < network.errorLimit;
            break; % Stopping criteria achieved! #break the loop!#
        end
        
        weightVector = vector_new;
        f_old = f_new;
        r_new = -1 * scg_calculateGradient(network, weightVector, trainingSetInput, trainingSetTarget); % TODO : calculate gradient
        
        success = true;
        lambda_bar = 0;
        
        if mod(iterIdx, network.totalWeightCount) == 0 
            gradient_new = r_new;
        else
            beta = (dot(r_new, r_new) - dot(r_new, r)) / phi;
            gradient_new = r_new + beta * gradient;
        end
        
        % decrease lambda according to comparison value
        if comparison > 0.75
            lambda = 0.5 * lambda;
        end
    else
        lambda_bar = lambda;
    end
    % increase lambda according to comparison value
    if comparison < 0.25
        lambda = 4 * lambda;
    end    
end % SCG main loop #end#


%% TEST
% pre-process test data set with chosen pre-process method.
if(network.preprocess == 1)
    testSetInput = scg_standardize(testSetInput);
elseif(network.preprocess == 2)
    testSetInput = scg_normalize(testSetInput);
end

% test dataset is applied to network
testResult = scg_test(network, weightVector, testSetInput);

% performance of test data set is obtained
[resultPercentage, resultMatrix, confMatrix] = scg_performance(testResult, testSetTarget);

resultPercentage
confMatrix;

% training dataset is applied to network
testResult_train = scg_test(network, weightVector, trainingSetInput);
[resultPercentage_t, resultMatrix_t, confMatrix_t] = scg_performance(testResult_train, trainingSetTarget);

resultPercentage_t;
confMatrix_t;

% get total of confusion matrix
confTotal = confMatrix + confMatrix_t;

confTotal

