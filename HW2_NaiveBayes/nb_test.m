function [Pred_nb] = nb_test(model, Xtest)
% use training set model to classify Xtest
% output prediction of labels

sizeX=size(Xtest);
test_num=sizeX(1);
word_num=sizeX(2);
% init output vector
Pred_nb(1:test_num,1)=-1;
% prior probabilities
pri0=model(3,1);
pri1=model(3,2);

for i=1:test_num
    L0=0;
    L1=0;
    % compute multinomial probability
    for j=1:word_num
        L0=L0+Xtest(i,j).*log(model(1,j));
        L1=L1+Xtest(i,j).*log(model(2,j));
    end
    L0=L0+log(pri0);
    L1=L1+log(pri1);
    % find the max of L
    if L0 >= L1
        Pred_nb(i)=0;
    else
        Pred_nb(i)=1;
    end
end

end
