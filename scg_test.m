% performs test with given trained network and test data set
function [ estimatedTestOutput ] = scg_test( network, trainedWeightVector, testDataSet )

    % Reshape vector to weights as in the network structure            
    [hiddenLayerWeights, outputLayerWeights] = scg_reshapeWeightsMatrix(network, trainedWeightVector);
    
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
    
    % calculate sum parts of hidden node
    sumOfOutputNode = activateOfHiddenNode * outputLayerWeights(2:end,:) + outputBiasMatrix;
    
    % Apply sum to softmax function
    activateOfOutputNode = scg_softmaxFunction(sumOfOutputNode, false);

    % Collect forward pass result
    estimatedTestOutput = activateOfOutputNode;
end

