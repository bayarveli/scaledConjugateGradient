% calculates success rate and confusion matrix
function [ classificationPercentage, resultMatrix, confMatrix ] = scg_performance( testOutput, targetSet )
    [rowCount, colCount] = size(testOutput);
    
    resultMatrix = zeros(rowCount, colCount);
    successClass = 0;
    
    for i=1:rowCount
        for j=1:colCount
            if(max(testOutput(i,:)) == testOutput(i,j))
                resultMatrix(i,j) = 1;
            else
                resultMatrix(i,j) = 0;
            end
        end
        
        if(resultMatrix(i,:) == targetSet(i,:))
            successClass = successClass + 1;           
        end
    end
    
    classificationPercentage = successClass / rowCount * 100;
    
    % Confusion matrix
    confMatrix = zeros(colCount,colCount);

    misClassX = 1;
    misClassY = 1;

    for i=1:rowCount
        % true classified
        if(resultMatrix(i,:) == targetSet(i,:))
            for j=1:colCount
                if(max(resultMatrix(i,:)) == resultMatrix(i,j))
                    confMatrix(j,j) = confMatrix(j,j) + 1;
                end            
            end
        % false classified
        else
            resultMatrix(i,:);
            targetSet(i,:);
            for j=1:colCount
                if(max(targetSet(i,:)) == targetSet(i,j))
                    misClassY = j;
                end
            end
            for j=1:colCount
                if(max(resultMatrix(i,:)) == resultMatrix(i,j))
                    misClassX = j;
                end
            end
            confMatrix(misClassX,misClassY) = confMatrix(misClassX,misClassY) + 1;
        end       
    end
end

