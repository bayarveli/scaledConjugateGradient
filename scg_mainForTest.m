% SCG implementation in function version, it is used for test purposes
function [ testResult ] = scg_main_test( trainingSetPercentage, testSetPercentage, preprocessType, hiddenNodeNumber)
    
    format long
    %% DATASET DEFINITION
    load('SamsungIMUAllData.mat');

    allDataset = scg_mixDataset(SamsungIMUAllData, 320);

    [trainingSet, testSet] = scg_divideDataset(allDataset, trainingSetPercentage, testSetPercentage);

    trainingSetInput = trainingSet(:,1:18);
    trainingSetTarget = trainingSet(:,19:27);

    testSetInput = testSet(:,1:18);
    testSetTarget = testSet(:,19:27);

    [trRowCount trColCount] = size(trainingSetInput);
    %% INITIALIZE NETWORK

    % WARNING! : Network settings is valid for 3-layer networks which are
    % fixed as 1 Input Layer, 1 Hidden Layer and 1 Output Layer. In each layer,
    % the number of nodes can be changed according to problem.
    networkSettings.inputNodeCount = 18;
    networkSettings.hiddenNodeCount = hiddenNodeNumber;
    networkSettings.outputNodeCount = 9;
    networkSettings.weightUpperLimit = 0.1;
    networkSettings.weightLowerLimit = -0.1;
    networkSettings.initialBiasValue = 0.0;
    networkSettings.maxIteration = 1000;
    networkSettings.errorLimit = 1.e-2;
    networkSettings.sampleCount = trRowCount; % sample count in training set
    networkSettings.preprocess = preprocessType; % 1 : Standardize, 2 : Normalize

    % Neural network is initialized with given settings
    network = scg_initializeNetwork(networkSettings);

    %% PREPROCESS
    if(network.preprocess == 1)
        trainingSetInput = scg_standardize(trainingSetInput);
    elseif(network.preprocess == 2)
        trainingSetInput = scg_normalize(trainingSetInput);
    end

    %% TRAINING
    % Scaled Conjugate Gradient (SCG) Algorithm is used for training algorithm.

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
        f_old = scg_calculateError(network, weightVector, trainingSetInput, trainingSetTarget);
        f_new = scg_calculateError(network, vector_new, trainingSetInput, trainingSetTarget);

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

            if comparison > 0.75
                lambda = 0.5 * lambda;
            end
        else
            lambda_bar = lambda;
        end

        if comparison < 0.25
            lambda = 4 * lambda;
        end    
    end % SCG main loop #end#

    %% TEST
    if(network.preprocess == 1)
        testSetInput = scg_standardize(testSetInput);
    elseif(network.preprocess == 2)
        testSetInput = scg_normalize(testSetInput);
    end

    testResult = scg_test(network, weightVector, testSetInput);

    resultPercentage = scg_performance(testResult, testSetTarget);
    
    testResult = [trainingSetPercentage testSetPercentage preprocessType hiddenNodeNumber resultPercentage iterIdx];
end

