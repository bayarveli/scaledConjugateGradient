%% Configuration test
% is conducted for chosing best parameter set for network
clc
clear

% Dataset division is applied from 50% to 90% with 10% step size
trainPercentage = 50:10:90;
% Hidden node count is changes from 8 to 64 with 8 step size
hiddenNodeCount = 8:8:64;
% preprocess options
preprocessEnum = [1 2];

resultIdx = 1;

for testTrainIdx = 1:length(trainPercentage)
    for preprocessIdx = 1:length(preprocessEnum)
        for hiddenIdx = 1:length(hiddenNodeCount)
 
            % apply related test parameters to the SCG implementation
            iterResult = scg_main_test(trainPercentage(testTrainIdx), ...
                                                      (100 - trainPercentage(testTrainIdx)), ...
                                                      preprocessEnum(preprocessIdx), ...
                                                      hiddenNodeCount(hiddenIdx));
            resultMatrix(resultIdx,:) = iterResult;
            resultIdx = resultIdx + 1;
        end
    end
end