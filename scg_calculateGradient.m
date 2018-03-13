% calculated gradient
function [ calculatedGradient ] = scg_calculateGradient( network, weightVector, trainingDataSet, trainingTargetSet)
    % #start# Forward pass
    % Reshape vector to weights as in the network structure            
    [hiddenLayerWeights, outputLayerWeights] = scg_reshapeWeightsMatrix(network, weightVector);
    
    % Apply all training dataset (inputs) to compute hidden layer sum
    [rowCount, colCount] = size(trainingDataSet);
    
    for i=1:rowCount
        hiddenBiasMatrix(i,:) = hiddenLayerWeights(1,:);
    end
    
    sumOfHiddenNode = trainingDataSet * hiddenLayerWeights(2:end,:) + hiddenBiasMatrix;
    % Apply sum to sigmoid activation function    
    activateOfHiddenNode = scg_sigmoidFunction(sumOfHiddenNode, false);
    % Take derivative of sigmoid activation function # will be used for
    % backpropagation
    activateDerivativeOfHiddenNode = scg_sigmoidFunction(sumOfHiddenNode, true);
    
    % Apply hidden layer results to output layer
    [rowCount, colCount] = size(activateOfHiddenNode);
    
    for i=1:rowCount
        outputBiasMatrix(i,:) = outputLayerWeights(1,:);
    end
    
    sumOfOutputNode = activateOfHiddenNode * outputLayerWeights(2:end,:) + outputBiasMatrix;
    % Apply sum to softmax function
    activateOfOutputNode = scg_softmaxFunction(sumOfOutputNode, false);
    % Take derivative of softmax activation function # will be used for
    % backpropagation
    activateDerivativeOfOutputNode = scg_softmaxFunction(sumOfOutputNode, true);
    % Collect forward pass result
    estimatedOutput = activateOfOutputNode;
    
    resultOfCostFunction = scg_softmaxCategoricalCrossEntropyCost(estimatedOutput, trainingTargetSet, true);
    
    % #start# Backward pass (use backpropagation)
    % 
    delta = resultOfCostFunction' .* activateDerivativeOfOutputNode;
    %
    [rowCount, colCount] = size(activateOfHiddenNode);
    hiddenNodeOutputWithBias = zeros(rowCount, (colCount + 1));
    for i=1:rowCount
        hiddenNodeOutputWithBias(i,:) = [1 activateOfHiddenNode(i,:)];
    end

    %
    deltaOutputLayer = delta * hiddenNodeOutputWithBias ./ network.sampleCount;
    % Collect output layer weights delta values
    deltaOutputLayer = deltaOutputLayer';
    
    %
    deltaFromOutputToHidden = outputLayerWeights(2:end,:) * delta;
    %
    delta = deltaFromOutputToHidden .* activateDerivativeOfHiddenNode;
    %
    [rowCount, colCount] = size(trainingDataSet);
    inputNodeOutputWithBias = zeros(rowCount, (colCount + 1));
    for i=1:rowCount
        inputNodeOutputWithBias(i,:) = [1 trainingDataSet(i,:)];
    end
    
    %
    deltaHiddenLayer = delta * inputNodeOutputWithBias ./ network.sampleCount;
    % Collect hidden layer weights delta values
    deltaHiddenLayer = deltaHiddenLayer';
    
    % Reshape delta matrices to delta vectors and combine them as delta
    % gradient vector
    [rowIdx, colIdx] = size(deltaHiddenLayer);
    deltaHiddenVector = zeros(1,(rowIdx * colIdx));
    outIdx = 1;
    for i=1:rowIdx
        for j=1:colIdx
            deltaHiddenVector(outIdx) = deltaHiddenLayer(i,j);
            outIdx = outIdx + 1;
        end
    end
    
    [rowIdx, colIdx] = size(deltaOutputLayer);
    deltaOutputVector = zeros(1,(rowIdx * colIdx));
    outIdx = 1;
    for i=1:rowIdx
        for j=1:colIdx
            deltaOutputVector(outIdx) = deltaOutputLayer(i,j);
            outIdx = outIdx + 1;
        end
    end
    % return calculated gradient vector
    calculatedGradient = [deltaHiddenVector deltaOutputVector];  
end

