function [preds] = nnPredictClassification(X, theta, output_size, opt)
%NNPREDICTCLASSIFICATION Predict the classes of each x_i
%
% function [preds] = nnPredictClassification(X, theta, output_size, opt)
%
%  X           - m x n design matrix
%  theta       - vector of all NN parameters
%  output_size - number of units in output layer.
%  opt         - NN parameters
% 
%  preds - 1 x m vector of predictions. Each element should be in {1,...,K}
%          assuming there are K classes.
% 

    % hint: the following line might be helpful.
    
    probabilities = nnComputeActivations(theta, X, output_size, opt);
    
    %% Compute the classes of each example in X.
    
    % NOT YET IMPLEMENTED %
    
    %% BEGIN SOLUTION
    [~, preds]=max(probabilities,[],1);
    %% END SOLUTION
    
end
