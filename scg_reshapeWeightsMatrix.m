% reshapes weights vector to weights matrices
function [ hiddenLayerWeights, outputLayerWeights ] = scg_reshapeWeightsMatrix( network, weightVector )
    hiddenWeightsVector = weightVector(1:network.hiddenWeightCount);
    outputWeightsVector = weightVector((network.hiddenWeightCount + 1):end);
    
    [hiddenWeightMatrixRow, hiddenWeightMatrixColumn] = size(network.hiddenLayerWeights);
    [outputWeightMatrixRow, outputWeightMatrixColumn] = size(network.outputLayerWeights);
    
    rowIdx = 1;
    colIdx = 1;
    for i=1:network.hiddenWeightCount
        hiddenLayerWeights(rowIdx, colIdx) = hiddenWeightsVector(i);
        
        colIdx = colIdx + 1;
        if mod(i,hiddenWeightMatrixColumn) == 0
            rowIdx = rowIdx + 1;
            colIdx = 1;
        end
    end
    
    rowIdx = 1;
    colIdx = 1;
    for i=1:network.outputWeightCount
        outputLayerWeights(rowIdx, colIdx) = outputWeightsVector(i);
        
        colIdx = colIdx + 1;
        if mod(i,outputWeightMatrixColumn) == 0
            rowIdx = rowIdx + 1;
            colIdx = 1;
        end
    end
end

