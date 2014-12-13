function theta = trainLR(X, y, lambda)
% TRAINLR  Fits a binary logistic regression model, with intercept.
%         theta = trainLR(X, y) fits a logistic regression model for
%         binary response y and data matrix X.
%
%         X - m x n design matrix of m observations with n features.
%         y - m x 1 vector of labels, either 0 or 1.
%         lambda - regularization term.
%
%         theta - (n+1) x 1 vector of predicted weights.

    [m, n] = size(X);
    theta = randn(n+1,1)*0.01; % initialize the weights randomly.
    
    % First, let's check that your J function's gradient and costs agree.
    A = randn(30, 4);
    A(1, :) = 1;
    B = rand(30, 1) < 0.5;
    checkGradient(@(x) costLR(A, B, x, lambda), randn(4,1));
    
    % Now let's check that your minimization function works.
    assert(abs(minimize(@quad, randn()) - 1) < 1e-2);
    
    % Note: you can swap the order of the checks above if you wish to 
    % implement gradient descent first.
    
    % Note: The logistic regression cost function is convex, which means
    % that if we run gradient descent or a similar method we are guaranteed
    % to approach the global minimum. With L2 regularization, we are
    % further guaranteed that there will be a unique global minimum.
    % Therefore, you should not worry about getting a different set of
    % thetas from us.
    
    % Write your solution below to compute the optimal thetas
    % (Our solution is ~2 lines):
    
    %% BEGIN SOLUTION
    X=[ones(m,1) X];
    yTemp=(y==1);
    theta = minimize(@(theta)(costLR(X,yTemp,theta,lambda)),theta);
    %% END SOLUTION
    
end

function [y, dy] = quad(x)
    y = (x - 1).^2;
    dy = 2*(x - 1);
end

function checkGradient(costfn, theta)
% CHECKGRADIENT  Checks your gradient against finite differences gradient
%               computed from function evaluation. Prints a warning if
%               L2 norm difference (sum of squares of differences) is
%               too large.
%
%   J should be a function of the form
%        [cost : scalar, grad : mx1 vector] = J(theta : mx1 vector)
%
    tol = 1e-6;
    n_dims = size(theta, 1);
    dx = 1e-5;
    id = eye(n_dims)*dx;
    grad_hat = zeros(size(theta));
    [~, grad] = costfn(theta); % get true analytical gradient.
    
    for i = 1:n_dims % now compute finite differences gradient.
        xp1 = costfn(theta + id(:, i));
        xm1 = costfn(theta - id(:, i));
        grad_hat(i) = (xp1 - xm1)/(2*dx); 
    end
    
    diff = norm(grad_hat - grad);
    
    if diff > tol
        fprintf(['Warning: Your gradients differ by %f. Your gradient'...
                 ' or cost function may be incorrect.\n'], diff);
    end
    assert(diff <= tol);
end
