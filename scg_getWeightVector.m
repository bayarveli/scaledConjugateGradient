% Returns weight vector converted from weights matrices
function [ weightVector ] = scg_getWeightVector( network )
    hiddenWeightsTranspoze = network.hiddenLayerWeights';
    hiddenWeightsVector = hiddenWeightsTranspoze(:);
    
    outputWeightsTranspoze = network.outputLayerWeights';
    outputWeightsVector = outputWeightsTranspoze(:);
    
    weightVector = [hiddenWeightsVector' outputWeightsVector'];
end

