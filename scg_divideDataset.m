% divides dataset into training and test sets with given percentage values
function [ trainingSet, testSet ] = scg_divideDataset( allSet, percentageTraining, percentageTesting )
    [rowCount colCount] = size(allSet);
    
    trainSize = rowCount * (percentageTraining / 100);
    testSize = rowCount * (percentageTesting / 100);
    
    trainingSet = allSet(1:round(trainSize),:);    
    testSet = allSet(round(trainSize + 1):round(trainSize + testSize), :);
end

