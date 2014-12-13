function [accuracy] = nb_run(Xtrain,Ytrain,Xtest,Ytest)
% a simple NB classifier
model=nb_train(Xtrain,Ytrain);
Pred_nb=nb_test(model,Xtest);
accuracy=mean(Pred_nb==Ytest);
save('Pred_nb.mat','Pred_nb');
end
