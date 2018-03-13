% Dataset rows are reordered according to target class value. Each row has
% different target class values listed recursively.
function [ mixSet ] = scg_mixDataset( trainSet, sampleSizePerClass )    
    % get row count of dataset
    [rowCount, colCount] = size(trainSet);
    % define counters for each target class
    counters = zeros(1, (rowCount/sampleSizePerClass));
    % initiaze first counter value
    counters(1) = 1;
    % calculate remained counters' values according to sample size per
    % class
    for i = 1:(length(counters)-1)
        counters(i+1) = i * sampleSizePerClass + 1;
    end

    startIdx = 1;
    % class set number 
    blockSize = length(counters);
    endIdx = startIdx * blockSize;
    % perform reorder operation
    for i = 1:sampleSizePerClass
        for j = 1:length(counters)
            block(j,:) = trainSet(counters(j),:);
            counters(j) = counters(j) + 1;
        end
        mixSet(startIdx:endIdx,:) = block;
        startIdx = endIdx + 1;
        endIdx = (i+1) * blockSize;
    end
end

