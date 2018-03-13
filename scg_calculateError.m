function [ calculatedError ] = scg_calculateError( network, weightVector, testDataSet, testTargetSet )

    % #start# Forward pass
    % Reshape vector to weights as in the network structure            
    [hiddenLayerWeights, outputLayerWeights] = scg_reshapeWeightsMatrix(network, weightVector);
    
    % Apply all training dataset (inputs) to compute hidden layer sum
    [rowCount, colCount] = size(testDataSet);
    
    for i=1:rowCount
        hiddenBiasMatrix(i,:) = hiddenLayerWeights(1,:);
    end
    
    sumOfHiddenNode = testDataSet * hiddenLayerWeights(2:end,:) + hiddenBiasMatrix;
    
    % Apply sum to sigmoid activation function    
    activateOfHiddenNode = scg_sigmoidFunction(sumOfHiddenNode, false);
    
    % Apply hidden layer results to output layer
    [rowCount, colCount] = size(activateOfHiddenNode);
    
    for i=1:rowCount
        outputBiasMatrix(i,:) = outputLayerWeights(1,:);
    end
    
    sumOfOutputNode = activateOfHiddenNode * outputLayerWeights(2:end,:) + outputBiasMatrix;
    
    % Apply sum to softmax function
    activateOfOutputNode = scg_softmaxFunction(sumOfOutputNode, false);

    % Collect forward pass result
    estimatedOutput = activateOfOutputNode;
    
    % perform cross entropy cost calculations
    calculatedError = scg_softmaxCategoricalCrossEntropyCost(estimatedOutput, testTargetSet, false);
end

