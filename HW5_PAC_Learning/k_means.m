
function [clusterCenters, clusterBelonging] = k_means(data, k, startPoints)

%--------------------------------------
%   YOUR CODE HERE
%--------------------------------------
point_num = size(data, 1);

if nargin < 3
    ip = randperm(point_num)';
    clusterCenters = data(ip(1:k,1),:);
else
    clusterCenters = startPoints;
end

clusterBelonging = zeros(point_num,1);

while 1
    d = computeDist(data, clusterCenters, k);
    [~,tempBelonging] = min(d,[],2);
    if tempBelonging == clusterBelonging
        break;
    else
        clusterBelonging = tempBelonging;
    end
    for i = 1:k
        belongToK = find(clusterBelonging == i);
        if belongToK
            clusterCenters(i,:) = mean(data((clusterBelonging==i),:),1);
        end
    end
end

end

function dist = computeDist(points, center, k)
    point_num = size(points,1);
    dist = zeros(point_num,k);
    for i=1:k
        centerOfK = repmat(center(i,:), point_num ,1);
        dist(:,i) = sqrt(sum(((points-centerOfK).^2), 2));
    end
end
