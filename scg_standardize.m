%% standardize dataset
function [ preprocessedTrainingData ] = scg_standardize( trainingData )
    
    [rowCount, colCount] = size(trainingData);
    
    % initialize mean and standart deviation vectors
    meanVector = zeros(1,colCount);
    stdVector = zeros(1,colCount);
    
    % calculate mean and standard deviation for each column
    for i=1:colCount
        meanVector(i) = mean(trainingData(:,i));
        stdVector(i) = std(trainingData(:,i),1); % population standart deviation, Flag = 1
    end

    % save each row standardised
    for i=1:rowCount
        for j=1:colCount
            preprocessedTrainingData(i,j) = (trainingData(i,j) - meanVector(j)) / stdVector(j);
        end
    end
        
end

