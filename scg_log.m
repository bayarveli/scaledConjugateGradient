% takes logarithm of each value in given matrix
function [ logOfMatrix] = scg_log( matrix )
    [rowIdx, colIdx] = size(matrix);
    
    for i=1:rowIdx
        for j=1:colIdx
            logOfMatrix(i,j) = log(matrix(i,j));
        end
    end
end

