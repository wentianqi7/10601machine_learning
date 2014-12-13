function act = nnComputeActivations(theta, X, output_size, opt)
%NNCOMPUTEACTIVATIONS  Compute the activations from the last layer of the
%                     neural network.
%
% function act = nnComputeActivations(theta, X, output_size, opt)
%
% theta       - learned parameter vector of all weights in the NN.
% X           - m x n design matrix. Note that in this case we will not
%               add a bias vector to the X's directly, since we are not
%               going to regularize the biases.
% layer_sizes - size of hidden layers and output layer.
% opt         - NN options
%
% act - activations of the output layer.
%
% Example usage:
%  acts = nnComputeActivations(theta, X, layers, opt);
%  acts = nnComputeActivations(theta, X, [hidsizes; outsize], opt);
%
    visible_size = size(X, 2);
    layer_sizes = [visible_size; opt.hidden_sizes; output_size];
    
    [Ws, bs] = unflattenParameters(theta, layer_sizes);
    
    % You may find the following variable helpful.
    
    W1 = Ws{1};
    W2 = Ws{2};
    b1 = bs{1};
    b2 = bs{2};
    
    %% Compute the activations of the output layer. Our solution is approx 
    %  10 lines.
    
    % NOT YET IMPLEMENTED %
    
    %% BEGIN SOLUTION
    m=size(X,1);
    z2=W1*X'+repmat(b1,1,m);
    a2=sigmoid(z2);
    z3=W2*a2+repmat(b2,1,m);
    a3=sigmoid(z3);
    act = a3;
    %% END SOLUTION
end

function y = sigmoid(x)
    y = 1./(1.+exp(-x));
end
