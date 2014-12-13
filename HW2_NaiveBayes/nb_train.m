function [model] = nb_train(Xtrain, Ytrain)
% train the NB classifier from feature set Xtrain and label set Ytrain
sizeX = size(Xtrain);
document_num = sizeX(1);
word_num = sizeX(2);
% init the output
% store Pr(W=j|Y=0) in the first line
model(1,1:word_num) = 0;
% store Pr(W=j|Y=1) in the second line
model(2,1:word_num) = 0;
% count of labels
c0 = 0;
c1 = 0;

for i = 1:document_num
    % if label is 0, update the value of Pr(W=j|Y=0)
    if Ytrain(i) == 0
        c0=c0 + 1;
        for j=1:word_num
            model(1,j) = model(1,j) + Xtrain(i,j);
        end
    % if label is 1, update the value of Pr(W=j|Y=1)
    else
        c1 = c1 + 1;
        for j = 1:word_num 
            model(2,j) = model(2,j) + Xtrain(i,j);
        end
    end
end

% additive smoothing
model(1,:)=(model(1,:)+1)./(sum(model(1,:))+word_num);
model(2,:)=(model(2,:)+1)./(sum(model(2,:))+word_num);
% store prior probabilities in the third line of model
model(3,1)=(c0+1)./(document_num+2);
model(3,2)=(c1+1)./(document_num+2);

end
