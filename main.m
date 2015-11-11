function [] = main()    
    addpath('library');
    percTrain = .8;
    percLabeledTrain = .05;
    [X,Y] = loadData();    
    isTrain = splitData(Y,percTrain);
    Xtrain = X(isTrain,:);
    Ytrain = Y(isTrain,:);
    
    isLabeledTrain = splitData(Ytrain,percLabeledTrain);
    numIterations = 10;
    
    %weights used for weighted cross validation
    sampleWeights = zeros(size(Ytrain));
    
    %Initially labeled data selected with uniform probability
    sampleWeights(isLabeledTrain) = 1/sum(isLabeledTrain);
    
    tab = sprintf('\t');
    for itr=1:numIterations
        Ycurr = Ytrain;
        Ycurr(~isLabeledTrain) = nan;
        [Ypred,beta] = trainAndTest(Xtrain,Ycurr);
        scores = uncertaintyScores(Ypred);
        
        %Construct sampling distribution and sample from it
        samplingDistribution = scores / sum(scores);
        i = discretesample(samplingDistribution, 1);
        sampleWeights(i) = samplingDistribution(i);
        
        isLabeledTrain(i) = true;       
        Ycurr(i) = Ytrain(i);
        
        %LOO estimates used for cross validation
        LOOestimates = getLOOEstimates(X(isLabeledTrain,:),Ycurr(isLabeledTrain));
        
        %Get estimates using Cross Validation, Weighted Cross Validation
        %and Our Method
        [CV,WCV,CVN] = getCVEstimates(LOOestimates,Y(isLabeledTrain),sampleWeights(isLabeledTrain));
        display(['Iteration: ' num2str(itr)]);
        display([tab 'CV: ' num2str(CV)]);
        display([tab 'WCV: ' num2str(WCV)]);
        display([tab 'CVN: ' num2str(CVN)]); 
        [Ypred] = predict(X(~isTrain,:),beta);
        display([tab 'True Error: ' num2str(mean(round(Ypred) == Y(~isTrain)))]);
    end
end

%Get CV Estimates:
%CV: Standard Cross Validation
%WCV: Weighted Cross Validation
%CVN: Weighted Cross Validation with Normalization (our approach);
function [CV,WCV,CVN] = getCVEstimates(Ypred,Y,weights)
    CV = mean(Ypred == Y);
    WCV = sum((Ypred == Y) .* weights);
    CVN = sum((Ypred == Y) .* (weights/sum(weights)));
end

%Simple active scoring scheme - measure how far prediction is from uniform
%distribution
function [scores] = uncertaintyScores(Ypred)
    scores = abs(Ypred - .5);
end

%Stratified split of data into [percTrain (1-percTrain)]
function [isTrain] = splitData(Y,percTrain)
    isTrain = false(size(Y));
    for idx=unique(Y)'
        hasLabel = find(Y == idx);
        n = length(hasLabel);
        perm = randperm(n,ceil(n*percTrain));
        isTrain(hasLabel(perm)) = true;
    end
end

%Get Leave-one-out estimates.  To speed this up you can use other types of
%estimates (e.g. create k-folds, make estimates on each fold when training
%on other folds)s
function [LOO] = getLOOEstimates(X,Y)
    LOO = zeros(size(Y));
    for i=1:length(Y)
        Yi = Y;
        Yi(i) = nan;
        Ypred = trainAndTest(X,Yi);
        LOO(i) = Ypred(i);
    end
    LOO = round(LOO);
end

%Train and make predictions on X and Y using l1 regularized Logistic
%Regression
function [Ypred,beta] = trainAndTest(X,Y)
    I = ~isnan(Y);
    Yt = Y(I);
    Xt = X(I,:);
    [beta,fitinfo] = lassoglm(Xt,Yt,'binomial','Lambda',.001);
    beta = [fitinfo.Intercept ; beta];
    Ypred = predict(X,beta);
end

function [Ypred] = predict(X,beta)
    Ypred = glmval(beta,X,'logit','constant','on');
end

%Load ionosphere data
function [X,Y] = loadData(file)
    file = 'ionosphere.mat';
    data = load(file);
    
    %Necessary for ionosphere data
    X = data.X(:,3:end);
    Yord = ordinal(data.Y);
    Y = double(Yord);
    I = Y == 1 | Y == 2;
    X = X(I,:);
    Y = Y(I);
    Y(Y == 2) = 0;
end