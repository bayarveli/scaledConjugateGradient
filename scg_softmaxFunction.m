% calculates softmax function with derivative option
function [ activationOutput ] = scg_softmaxFunction( weightMatrix, takeDerivative)
    %
    [rowIdx, colIdx] = size(weightMatrix);
    
    interOutput = zeros(rowIdx, colIdx);
    sumOfRowExponential = zeros(rowIdx,1);
    
    for i=1:rowIdx
        for j=1:colIdx
            interOutput(i,j) = exp(weightMatrix(i,j));
        end
        sumOfRowExponential(i) = sum(interOutput(i,:));
    end
    
    scaledOutput = zeros(rowIdx, colIdx);
    
    for i=1:rowIdx
        for j=1:colIdx
            scaledOutput(i,j) = interOutput(i,j) / sumOfRowExponential(i);
        end
    end
    
    if takeDerivative
        activationOutput = ones(rowIdx, colIdx);
        activationOutput = activationOutput';
    else
        activationOutput = scaledOutput;
    end   
end

