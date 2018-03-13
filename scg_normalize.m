%% normalize dataset
function [ normalizedData ] = scg_normalize( rawData )
    % get column count of dataset
    [rowCount, colCount] = size(rawData);
    
    for i=1:colCount
        % calculate minimum value of related column
        minVal = min(rawData(:,i));
        % calculate maximum value of related column
        maxVal = max(rawData(:,i));
        % get center value
        center = (maxVal + minVal) / 2;
        % get range value
        range = (maxVal - minVal) / 2;
        % save each row value normalised.
        normalizedData(:,i) = (rawData(:,i) - center) / range;
    end
end

