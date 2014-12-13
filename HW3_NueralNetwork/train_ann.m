function [X_test, Y_test] = train_ann()
    %% Tricks applied
    % 1. Batch Gradient Descent: learning rate = 0.005
    % 2. 1 hidden layer neural network
    % 3. symmetry breaking: initialize to Gaussian with 0 mean and 0.005
    % variance
    % 4. Activation Function: sigmoid
    %% Tricks end
    
    %% BEGIN SOLUTION
    % You are free to modify any code in this section. Just make sure to 
    % store your weights
    load('digits.mat');
    [m,ftr] = size(XTrain);
    
    % separate data
    trSize = m*0.6;
    vSize = m*0.2;
    X_verify = XTrain(trSize+1:trSize+vSize,:);
    X_test = XTrain(trSize+vSize+1:m,:);
    X_train = XTrain(1:trSize,:);
    Y_verify = yTrain(trSize+1:trSize+vSize,1);
    Y_test = yTrain(trSize+vSize+1:m,1);
    Y_train = yTrain(1:trSize,1);
    
    % init parameters
    num_classes = 10;
    opt.hidden_sizes = 100;
    opt.lambda = 0.1;
    opt.MaxIter = 10000;
    
    % initialize weights
    W1 = normrnd(0,0.005,[opt.hidden_sizes ftr+1]);
    W2 = normrnd(0,0.005,[num_classes opt.hidden_sizes]);
    b1 = normrnd(0,0.005,[opt.hidden_sizes 1]);
    b2 = normrnd(0,0.005,[num_classes 1]);
%     W1 = ones(opt.hidden_sizes, ftr+1)/2;
%     W2 = ones(num_classes,opt.hidden_sizes)/2;
%     b1 = zeros(opt.hidden_sizes,1);
%     b2 = zeros(num_classes,1);
    theta{1} = W1;
    theta{2} = W2;
    theta{3} = b1;
    theta{4} = b2;
    
    % store initial weights
    save('ini_weights.mat','W1','W2','b1','b2');
    % put your training code here
    Y_train_temp = full(sparse(Y_train+1, 1:trSize, 1));
    X_train = [ones(trSize,1) X_train];
    op_theta = minimize(@(x) costNN(X_train, Y_train_temp, x, opt), theta, opt);
    % save your final weights
    W1 = op_theta{1};
    W2 = op_theta{2};
    b1 = op_theta{3};
    b2 = op_theta{4};
    save('weights.mat','W1','W2','b1','b2');
    
    % test training result with verify group
    X_verify = [ones(vSize,1) X_verify];
    trainpreds = predict(X_train, op_theta);
    preds = predict(X_verify, op_theta);
    
    traincorrects = trainpreds-1 == (Y_train');
    trainaccuracy = 100*mean(traincorrects);
    fprintf('Train accuracy: %.3f%%\n', trainaccuracy);

    corrects = preds-1 == (Y_verify');
    accuracy = 100*mean(corrects);
    fprintf('Verify accuracy: %.3f%%\n', accuracy);
    %% END SOLUTION
end

function [cost, grad] = costNN(X, y, theta, opt)
    feature_size = size(X, 2);
    hidden_size = opt.hidden_sizes;
    output_size = size(y, 1);

    W1 = theta{1}; % weights from layer 1 to layer 2.
    W2 = theta{2}; % weights from layer 2 to layer 3.
    b1 = theta{3}; % biases to layer 2.
    b2 = theta{4}; % biases to layer 3.
    
    W1grad = zeros(hidden_size, feature_size);
    W2grad = zeros(output_size, hidden_size);
    b1grad = zeros(hidden_size, 1);
    b2grad = zeros(output_size, 1);
    
    %% compute cost and grad
    acts=nnComputeActivations(theta,X);
    a2 = acts{1};
    a3 = acts{2};
    % compute cost
    tempCost=((a3-y).^2)./2;
    sumCost=sum(tempCost(:));
    W12=W1.^2;
    W22=W2.^2;
    cost=sumCost+(opt.lambda/2)*(sum(W12(:))+sum(W22(:)));
    % compute grad
    er3=(a3-y).*a3.*(1-a3);
    er2=W2'*er3.*a2.*(1-a2);
    W1grad=opt.lambda*W1+er2*X;
    W2grad=opt.lambda*W2+er3*a2';
    b1grad=sum(er2,2);
    b2grad=sum(er3,2);
    %% end computation
    grad{1} = W1grad;
    grad{2} = W2grad;
    grad{3} = b1grad;
    grad{4} = b2grad;
end

function acts = nnComputeActivations(theta, X)
    W1 = theta{1};
    W2 = theta{2};
    b1 = theta{3};
    b2 = theta{4};
    
    %% compute activations for output layer
    m=size(X,1);
    z2=W1*X'+repmat(b1,1,m);
    a2=actFunc(z2);
    z3=W2*a2+repmat(b2,1,m);
    a3=actFunc(z3);
    acts{1} = a2;
    acts{2} = a3;
    %% end computation
end

function theta = minimize(f, init_theta, opt)
    tol = 1e-5;       
    alpha = 0.005;
    alpha_decay = 0.99998;
    %% BEGIN SOLUTION (GRADIENT DESCENT)
    [init_cost, init_grad]=f(init_theta);
    for i=1:opt.MaxIter
        theta{1} = init_theta{1}-alpha*init_grad{1};
        theta{2} = init_theta{2}-alpha*init_grad{2};
        theta{3} = init_theta{3}-alpha*init_grad{3};
        theta{4} = init_theta{4}-alpha*init_grad{4};
        [cost, grad]=f(theta);
        if abs(cost-init_cost)<tol
            break;
        else
            abs(cost-init_cost)
            init_cost=cost;
            init_grad=grad;
            init_theta=theta;
            alpha=alpha*alpha_decay;
        end
    end
end

function [preds] = predict(X, theta)
    prob = nnComputeActivations(theta, X);
    [~, preds]=max(prob{2},[],1);
end

function y = actFunc(x)
    y = 1./(1.+exp(-x));
    %y = (exp(x)-exp(-x))./(exp(x)+exp(-x));
end
