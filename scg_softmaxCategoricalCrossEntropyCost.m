% calculated cross entropy cost with derivative option
function [ resultOfCost ] = scg_softmaxCategoricalCrossEntropyCost( estimatedOutput, targetOutput, takeDerivative)

    epsilon = 1.e-11;
    
    estimatedOutput = scg_preventOverflow(estimatedOutput, epsilon, 1-epsilon);

    if takeDerivative
        resultOfCost = estimatedOutput - targetOutput;
    else
        logOfEstimatedOutput = scg_log(estimatedOutput);
        interOutput = targetOutput .* logOfEstimatedOutput;
        
        [rowIdx, colIdx] = size(interOutput);        
        sumOfError = zeros(rowIdx,1);
        
        for i=1:rowIdx
            sumOfError(i) = -1 * sum(interOutput(i,:));
        end
        
        resultOfCost = mean(sumOfError);
    end
end
    


