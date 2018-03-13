% limits values of a matrix with given minimum and maximum value
function [ matrix ] = scg_preventOverflow( matrix, minValue, maxValue)
    [rowIdx, colIdx] = size(matrix);
    
    for i=1:rowIdx
        for j=1:colIdx
            if (matrix(i,j) < minValue)
                matrix(i,j) = minValue;
            elseif (matrix(i,j) > maxValue)
                matrix(i,j) = maxValue;
            end
        end
    end

end

