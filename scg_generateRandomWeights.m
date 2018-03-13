% generates random weight values between upper and lower limits given.
% adding bias is optional
function [ generatedWeightMatrix ] = scg_generateRandomWeights( weightMatrix, weightUpperLimit, weightLowerLimit, addBias, initialBias)
    % Generates randow weight values within given lower and upper limits.
    [rowCount, columnCount] = size(weightMatrix);
    % (b-a).*rand(1000,1) + a;
    generatedWeightMatrix = (weightUpperLimit - weightLowerLimit) * rand(rowCount, columnCount) + weightLowerLimit;
    
    bias_vector = zeros(1, rowCount);
    if(addBias)
        for i = 1:rowCount
            bias_vector(i) = initialBias;
        end
        generatedWeightMatrix = [bias_vector; generatedWeightMatrix'];
    end
end

