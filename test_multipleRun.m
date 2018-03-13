%% Multiple Run test
clc
clear

resultIdx = 1;

% 50 times run
for testTrainIdx = 1:50
    % apply fixed test parameters to the SCG implementation
    iterResult = scg_main_test(70, 30, 1, 32);
    resultMatrix(resultIdx,:) = iterResult;
    resultIdx = resultIdx + 1;
end