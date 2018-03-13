% Initializes network with given settings
function [ initialized_network ] = scg_initializeNetwork(network_settings)
    % Weight links from input nodes to hidden nodes without bias weights
    countInputToHiddenLinksWithoutBias = network_settings.inputNodeCount * network_settings.hiddenNodeCount;
    % Weight links included bias weights which count is same as hidden node count
    countInputToHiddenLinksWithBias = countInputToHiddenLinksWithoutBias + network_settings.hiddenNodeCount;
    
    % Weight links from hidden nodes to output nodes without bias weights
    countHiddenToOutputLinksWithoutBias = network_settings.hiddenNodeCount * network_settings.outputNodeCount;
    % Weight links included bias weights which count is same as output count
    countHiddenToOutputLinksWithBias = countHiddenToOutputLinksWithoutBias + network_settings.outputNodeCount;
    
    % set hidden weights count in the network # will be used reshape
    % weight vector
    network.hiddenWeightCount = countInputToHiddenLinksWithBias;
    
    % set hidden weights count in the network # will be used reshape
    % weight vector
    network.outputWeightCount = countHiddenToOutputLinksWithBias;
    
    % set total weights(links) count in the network
    network.totalWeightCount = countInputToHiddenLinksWithBias + countHiddenToOutputLinksWithBias;
    
    % initialize hidden layer weight matrice without bias
    network.hiddenLayerWeights = zeros(network_settings.hiddenNodeCount, network_settings.inputNodeCount);
    
    % initialize output layer weight matrice without bias
    network.outputLayerWeights = zeros(network_settings.outputNodeCount, network_settings.hiddenNodeCount);
    
    % populate hidden layer weights randomly
    network.hiddenLayerWeights = scg_generateRandomWeights(network.hiddenLayerWeights, ...
                                                           network_settings.weightUpperLimit, ...
                                                           network_settings.weightLowerLimit, ...
                                                           true, ...
                                                           network_settings.initialBiasValue);
    % populate output layer weights randomly
    network.outputLayerWeights = scg_generateRandomWeights(network.outputLayerWeights, ...
                                                           network_settings.weightUpperLimit, ...
                                                           network_settings.weightLowerLimit, ...
                                                           true, ...
                                                           network_settings.initialBiasValue);
    % maximum iteration count                                                   
    network.maxIteration = network_settings.maxIteration;
    % error limit as stopping criteria
    network.errorLimit = network_settings.errorLimit;
    % sample count
    network.sampleCount = network_settings.sampleCount;
    % preprocess type
    network.preprocess = network_settings.preprocess;
    % return initialized network
    initialized_network = network;
end

