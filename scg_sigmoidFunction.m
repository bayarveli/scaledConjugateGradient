% calculates log-sigmoid function with derivative option
function [ activationOutput ] = scg_sigmoidFunction( sumMatrix, takeDerivative)
    sumMatrix = scg_preventOverflow(sumMatrix, -500, 500);

    [rowIdx, colIdx] = size(sumMatrix);
    
    interOutput = zeros(rowIdx, colIdx);
    
    for i=1:rowIdx
        for j=1:colIdx
            interOutput(i,j) = 1 / (1 + exp(-sumMatrix(i,j)));
        end
    end
    
    if(takeDerivative)
        activationOutput = interOutput .* (1 - interOutput);
        activationOutput = activationOutput';
    else
        activationOutput = interOutput;
    end    
end

