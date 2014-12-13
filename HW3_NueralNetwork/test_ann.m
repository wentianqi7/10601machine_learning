function [y] = test_ann(XTest)
    %% BEGIN SOLUTION
    % You are free to modify any code in this section. Just make sure to 
    % load your trained weights from weights.mat
    load('weights.mat');
%     [m,~] = size(XTest);
%     h = actFunc([ones(m,1) XTest]*W1');
%     [~,y] = max(h*W2',[],2);
%     y = y-1;
    a3 = nnComputeActivations(W1,W2,b1,b2,XTest);
    [~,y] = max(a3,[],2);
    y=y-1;
    %% END SOLUTION
end

function a3 = nnComputeActivations(W1,W2,b1,b2, X)
    %% compute activations for output layer
    m=size(X,1);
    X = [ones(m,1) X];
    z2=W1*X'+repmat(b1,1,m);
    a2=actFunc(z2);
    z3=W2*a2+repmat(b2,1,m);
    a3=actFunc(z3);
    a3=a3';
    %% end computation
end

function y = actFunc(x)
    y = 1./(1.+exp(-x));
end
