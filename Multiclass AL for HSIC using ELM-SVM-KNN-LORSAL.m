% Author : Muhammad Ahmad (Ph.D.)
% Date   : 26/07/2020
% Email  : mahmad00@gmail.com
%% Clear and Close Everything
clc; clear; close all;
%% Warning off
warning('off','all');
%% Loading Dataset
load('Data');
%% Active Learning Classifiers
[OA, ~, kappa, Time] = My_AL(img, TrC, TeC, AL_Strtucture, NN, HN, Act_Fun, gt);
%% Ploting and Saving Accuracies
%% Overall Acuracy
figure(1)
set(gcf,'color','w');
set(gca, 'fontsize', 12, 'fontweight','bold')
hold on
plot(OA(:,1), '--s','MarkerSize', 8,...
        'MarkerEdgeColor','red', 'LineWidth', 2.5)
hold on
plot(OA(:,2), '-.o','MarkerSize', 8,...
        'MarkerEdgeColor','blue', 'LineWidth', 2.5)
hold on
plot(OA(:,3), ':*','MarkerSize', 8,...
        'MarkerEdgeColor','black', 'LineWidth', 2.5)
hold on
plot(OA(:,4), '-.+','MarkerSize', 8,...
        'MarkerEdgeColor','cyan', 'LineWidth', 2.5)
hold on
legend({'SVM', 'MLR', 'KNN', 'ELM'},'FontSize',12,...
        'FontWeight','bold','Location','southeast', 'color','k');
legend('boxoff'); grid on;
ylabel('Overall Accuracy','FontSize',12,'FontWeight','bold', 'color','k')
xlabel('Number of Iterations','FontSize',12,'FontWeight','bold', 'color','k')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(figure(1), 'Overall.png');
pause(1)
close all;
%% Kappa Accuracy
figure(1)
set(gcf,'color','w');
set(gca, 'fontsize', 10, 'fontweight','bold')
hold on
plot(kappa(:,1), '--s','MarkerSize', 8,...
        'MarkerEdgeColor','red', 'LineWidth', 2.5)
hold on
plot(kappa(:,2), '-.o','MarkerSize', 8,...
        'MarkerEdgeColor','blue', 'LineWidth', 2.5)
hold on
plot(kappa(:,3), ':*','MarkerSize', 8,...
        'MarkerEdgeColor','black', 'LineWidth', 2.5)
hold on
plot(kappa(:,4), '-.+','MarkerSize', 8,...
        'MarkerEdgeColor','cyan', 'LineWidth', 2.5)
hold on
legend({'SVM', 'MLR', 'GB', 'LB', 'KNN', 'ELM'},'FontSize',12,...
        'FontWeight','bold','Location','southeast', 'color','k');
legend('boxoff'); grid on;
ylabel('kappa','FontSize',12,'FontWeight','bold', 'color','k')
xlabel('Number of Iterations','FontSize',12,'FontWeight','bold', 'color','k')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(figure(1), 'kappa.png');
pause(1)
close all;
%% Ploting Time
figure(1)
set(gcf,'color','w');
set(gca, 'fontsize', 12, 'fontweight','bold')
hold on
plot(Time(:,1), '--s','MarkerSize', 8,...
        'MarkerEdgeColor','red', 'LineWidth', 2.5)
hold on
plot(Time(:,2), '-.o','MarkerSize', 8,...
        'MarkerEdgeColor','blue', 'LineWidth', 2.5)
hold on
plot(Time(:,3), ':*','MarkerSize', 8,...
        'MarkerEdgeColor','black', 'LineWidth', 2.5)
hold on
plot(Time(:,4), '--s','MarkerSize', 8,...
        'MarkerEdgeColor','yellow', 'LineWidth', 2.5)
hold on
legend({'SVM', 'MLR', 'KNN', 'ELM'},'FontSize',12,...
        'FontWeight','bold','Location','northwest', 'color','k');
legend('boxoff'); grid on;
ylabel('Time','FontSize',12,'FontWeight','bold', 'color','k')
xlabel('Number of Iterations','FontSize',12,'FontWeight','bold', 'color','k')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(figure(1), 'Time.png');
pause(1)
close all;
%% Internal Functions
function [OA, AA, kappa, Time] = My_AL(varargin)
%[OA, AA, kappa] = My_AL(varargin)
% Author : Muhammad Ahmad
% Date   : 10/10/2018
% Email  : mahmad00@gmail.com
% Reference: Multiclass Non-Randomized Spatial-Spectral Pool-based ...
%   Interactive Learning for Hyperspectral Image Classification
% For further information/queries please contact the authros
% Email: mahmad00@gmail.com
%% Compiling the Input
%% 1st Parameter is Hyperspectral Dataset
img = varargin{1};
if ~numel(img),error('No Hyperspectral Data');end
%% 2nd Parameter is Training Labels with Spatial Information
TrC = varargin{2};
if isempty(TrC)
    error('Please Include the Training Labels');
end
% uc = unique(TrC(2,:));
%% 3rd Parameter is Test Labels with Spatial Information
if nargin > 2
    TeC = varargin{3};
else
    fprintf('Test Labeles Not Available, Thus we used Training as Validation');
    TeC = TrC;
end
%% 4th Parameter is Active Learning strtucture
if nargin > 3
    AL_Strtucture = varargin{4};
else
    AL_Strtucture = [];
end
%% 5th Parameter is about the Fuzziness (High/Low)
if nargin > 4
    NN = varargin{5};
else
    NN = 20;
end
%% 6th Parameter is about the Fuzziness (High/Low)
if nargin > 5
    HN = varargin{6};
else
    HN = 20;
end
%% 7th Parameter is about the Fuzziness (High/Low)
if nargin > 6
    Act_Fun = varargin{7};
else
    Act_Fun = 'sin';
end
%% 8th Parameter is about the Fuzziness (High/Low)
if nargin > 7
    gt = varargin{8};
end
%% Multinomial Logistic Regression/LORSAL
[MLR_Time, MLR_Results] = My_MLR(img, TrC, TeC, AL_Strtucture, gt);
%% Support Vector Machine (SVM)-based Active Learning
[SVM_Time, SVM_Results] = My_SVM(img, TrC, TeC, AL_Strtucture, gt);
%% K Nearest Neigbors-based Active Learning
[KNN_Time, KNN_Results] = My_KNN(img, TrC, TeC, AL_Strtucture, NN, gt);
%% Extreme Learning Machine (ELM)-based Active Learning
[ELM_Time, ELM_Results] = My_ELM(img, TrC, TeC, AL_Strtucture, HN, Act_Fun, gt);
%% Computational Time
Time = [SVM_Time, MLR_Time, KNN_Time, ELM_Time];
%% Handle The Accuracies
%% Reshape Results
%% MLR
MLR_OA = MLR_Results.OA;
MLR_kappa = MLR_Results.kappa;
MLR_AA = MLR_Results.AA;
%% Sport Vector Machine (SVM)
SVM_OA = SVM_Results.OA;
SVM_kappa = SVM_Results.kappa;
SVM_AA = SVM_Results.AA;
%% K Nearset Neighbours (KNN)
KNN_OA = KNN_Results.OA;
KNN_kappa = KNN_Results.kappa;
KNN_AA = KNN_Results.AA;
%% Extreme Learning Machine (ELM)
ELM_OA = ELM_Results.OA;
ELM_kappa = ELM_Results.kappa;
ELM_AA = ELM_Results.AA;
%% Average, Overall, and kappa Accuracy to Return
OA = [SVM_OA' MLR_OA' KNN_OA' ELM_OA'];
AA = [SVM_AA' MLR_AA' KNN_AA' ELM_AA'];
kappa = [SVM_kappa' MLR_kappa' KNN_kappa' ELM_kappa'];
end
%% Internal Functions
%% LORSAL Classifier
function [time, MLR_Class_Results] = My_MLR(varargin)
% Author : Muhammad Ahmad
% Date   : 10/10/2018
% Email  : mahmad00@gmail.com
% For further information/queries please contact the authros
% Email: mahmad00@gmail.com

% This code partially taken from 
% % [1] J. Li, J. Bioucas-Dias and A. Plaza. Spectral-Spatial Classification 
% %  of Hyperspectral Data Using Loopy Belief Propagation and Active Learning. 
% %  IEEE Transactions on Geoscience and Remote Sensing. 2012. Accepted
% % [2] J. Li, J. Bioucas-Dias and A. Plaza. Hyperspectral Image Segmentation
% %  Using a New Bayesian Approach with Active Learning. IEEE Transactions on
% %  Geoscience and Remote Sensing.vol.49, no.10, pp.3947-3960, Oct. 2011.
% %  Copyright: Jun Li (jun@lx.it.pt) 
% %             Jos? Bioucas-Dias (bioucas@lx.it.pt)
% %             Antonio Plaza (aplaza@unex.es)
%% Compiling the Input
%% 1st Parameter is Hyperspectral Dataset
img = varargin{1};
img = img';
if ~numel(img),error('No Hyperspectral Data');end
%% 2nd Parameter is Training Labels with Spatial Information
TrC = varargin{2};
if isempty(TrC)
    error('Please Include the Training Labels');
end
uc = unique(TrC(2,:));
%% 3rd Parameter is Test Labels with Spatial Information
if nargin > 2
    TeC = varargin{3};
else
    fprintf('Test Labeles Not Available, Thus we used Training as Validation');
    TeC = TrC;
end
%% 4th Parameter is Active Learning strtucture
if nargin > 3
    AL_Strtucture = varargin{4};
else
    AL_Strtucture = [];
end
if isempty(AL_Strtucture)
    tot_sim = 1;
else
    tot_sim = AL_Strtucture.M/AL_Strtucture.h + 1;
end
%% 5th Parameter is Ground Truths
gt = varargin{5};
if ~numel(gt)
    error('No Hyperspectral Ground Truths');
end
%%
algorithm_parameters.lambda=0.001;
algorithm_parameters.beta = 0.5*algorithm_parameters.lambda;
algorithm_parameters.mu = 4;
%% Saving Test Samples Locations
TeC_Locations = cell(tot_sim, 1);
MLR_results = cell(tot_sim, 1);
time = zeros(tot_sim, 1);
%% Start Active Learning Process For Several Classifiers
for iter = 1: tot_sim
    tic;
    fprintf('LORSAL Active Selection %d \n', iter)
    Tr = img(:, TrC(1,:));      %% Training Set
    Te = img(:, TeC(1,:));      %% Training Set
    TeC_Locations{iter} = TeC;
%% MLR Classifier
    sigma = 0.8;
    %% build |x_i-x_j| matrix 
    nx = sum(Tr.^2);
    [X,Y] = meshgrid(nx);
    dist = X + Y - 2*Tr'*Tr; clear X Y
    scale = mean(dist(:));
    %% build design matrix (kernel) 
    K = exp(-dist/2/scale/sigma^2); clear dist
    %% set first line to one 
    K = [ones(1,size(Tr, 2)); K];
    learning_output = struct('scale',scale,'sigma',sigma);
    %% Train and Test LORSAL 
    [w, ~] = LORSAL(K, TrC(2,:), algorithm_parameters.lambda, algorithm_parameters.beta);
    %% Compute the MLR probabilites
    belief = mlr_probabilities(Te, Tr, w, learning_output);
    %% Compute the MLR results
    [~, MLR_results{iter}] = max(belief);
    [MLR_Class_Results.OA(iter), MLR_Class_Results.kappa(iter),...
        MLR_Class_Results.AA(iter)] = My_Accuracy(TeC(2,:), ...
            MLR_results{iter},(1:numel(uc)));
        %% Fuzziness Based Sample Catagorization
            MLR_Fuz = My_Fuzziness(belief');
            Pred = MLR_results{iter};
            Pred  = [Pred; AL_Strtucture.Candidate_Set];
            Pred = [MLR_Fuz'; Pred]';
            pre = sortrows(Pred, 4);
            [AB, ~] = find(pre(:,4) ~= 0);
            Pred = pre(AB:end,:);
            [A, ind] = sortrows(Pred, -1);
            [idx, ~] = find(A(:,4) ~= A(:,2));
            index_MLR_minME = ind(idx);
            if length(index_MLR_minME)>(AL_Strtucture.h)
                xp = index_MLR_minME(1 : AL_Strtucture.h)';
            else
                ind(idx) = [];
                index_MLR_minME = [index_MLR_minME' ind'];
                xp = index_MLR_minME(1 : AL_Strtucture.h)';
            end
            TrCNew = AL_Strtucture.Candidate_Set(:,xp);
            pre = sortrows(TrCNew', 2);
            [AB, ~] = find(pre(:,2) ~= 0);
            TrCNew = pre(AB:end,:)';
            TrC = [TrC, TrCNew];
            AL_Strtucture.Candidate_Set(:,xp) = [];
            TeC = AL_Strtucture.Candidate_Set;
%% End for Iter on AL
    time(iter,:) = toc; clear tic toc
end
end
%% LORSAL Classifier 
function [v,L] = LORSAL(x, y, lambda, beta, MMiter, w0)
% Sparse Multinomial Logistic Regression  
%  Implements a block Gauss Seidel  algorithm for fast solution of 
%  the SMLR introduced in Krishnapuram et. al, IEEE TPMI, 2005 (eq. 12)
%% Bregman weight
BR_iter = 1;
Bloc_iters = 1;
if nargin < 5
    MMiter = 200;
end
%% [d - space dimension, n-number of samples]
[d, n] = size(x);
%% number of classes 
m = max(y);
%% Auxiliar matrix to compute a bound for the logistic hessian
U = -1/2*(eye(m-1)-ones(m-1)/m); 
%% Convert y into binary information
Y = zeros(m,n);
for i=1:n
    Y(y(i),i) = 1;
end
%% Remove last line
Y=Y(1:m-1,:);
%% Build Rx
Rx = x*x';
%% Initialize w with ridge regression fitted to w'*x = 10 in the class
%% and w'*x = -10 outside the class
if (nargin < 6)
    alpha = 1e-5;
    w = (Rx+alpha*eye(d))\(x*(10*Y'-5*ones(size(Y'))));
else
    w = w0;
end
%% Do eigen-decomposition
[Uu,Du] = eig(U);
[Ur,Dr] = eig(Rx);
S = 1./(repmat(diag(Dr),1,m-1)*Du -beta*ones(d,m-1));
%% Bregman iterative scheme to compute w
%% initialize v (to impose the constraint w=v)
v = w;
%% initialize the Bregman vector b
b = 0*v;
%% MM iterations
for i=1:MMiter
%     fprintf('\n i = %d',i);
    %% compute the  multinomial distributions (one per sample)
    aux1 = exp(w'*x);
    aux2 = 1+sum(aux1,1);
    p =  aux1./repmat(aux2,m-1,1);
    %% compute log-likelihood
    L(i) = sum(sum(Y.*(w'*x),1) -log(aux2)) -lambda*sum(abs(w(:)))+ ...
        lambda*sum(abs(w(1,:)))  ;
    %% Compute derivative
    dg = Rx*w*U'-x*(Y-p)';
    %% Bregman iterations
    for k = 1:BR_iter
       for j=1:Bloc_iters
           %% update w
           z = dg-beta*(v+b);
           w = S.*(Ur'*z*Uu);
           w = Ur*w*Uu';
           %% update v
           v=wthresh(w-b,'s',lambda/beta);
       end
       %% Bregman factor
       b = b-(w-v);
    end
    beta = beta*1.05;
    S = 1./(repmat(diag(Dr),1,m-1)*Du -beta*ones(d,m-1));
end
end
%% Compute Probabilites
function p = mlr_probabilities(input,z,w,learning_output)
x=input;
clear input
[~, n] = size(x);
nz = sum(z.^2);
n1 = floor(n/80);
p = [];
for i = 1:79
    x1 = x(:,((i-1)*n1+1):n1*i);    
    if learning_output.scale == 0
        K1 = [ones(1,n1); x1];
    else
        nx1 = sum(x1.^2);
        [X1,Z1] = meshgrid(nx1,nz);
        clear nx1;
        dist1 = Z1-2*z'*x1+X1;
        K1 = exp(-dist1/2/learning_output.scale/learning_output.sigma^2);
        K1 = [ones(1,n1); K1];
    end
    p1 = mlogistic(w,K1);
    p = [p p1];
end
x1 = x(:,(79*n1+1):n);
clear x
if learning_output.scale == 0
    K1 = [ones(1,n-79*n1); x1];
else
    nx1 = sum(x1.^2);
    [X1,Z1] = meshgrid(nx1,nz);
    dist1 = Z1-2*z'*x1+X1;
    K1 = exp(-dist1/2/learning_output.scale/learning_output.sigma^2);
    K1 = [ones(1,n-79*n1); K1];
end
p1 = mlogistic(w,K1);
p = [p p1];
end
%% Compute Logistics 
function p = mlogistic(w, x)
%% compute the  multinomial distributions (one per sample)
m = size(w,2)+1;
aux = exp(w'*x);
p =  aux./repmat(1+sum(aux,1),m-1,1);
%% last class
p(m,:) = 1-sum(p,1);
end
%% SVM
function [time, SVM_Class_Results] = My_SVM(varargin)
%% Compiling the Input
%% 1st Parameter is Hyperspectral Dataset
img = varargin{1};
if ~numel(img),error('No Hyperspectral Data');end
%% 2nd Parameter is Training Labels with Spatial Information
TrC = varargin{2};
if isempty(TrC)
    error('Please Include the Training Labels');
end
uc = unique(TrC(2,:));
%% 3rd Parameter is Test Labels with Spatial Information
if nargin > 2
    TeC = varargin{3};
else
    fprintf('Test Labeles Not Available, Thus we used Training as Validation');
    TeC = TrC;
end
%% 4th Parameter is Active Learning strtucture
if nargin > 3
    AL_Strtucture = varargin{4};
else
    AL_Strtucture = [];
end
if isempty(AL_Strtucture)
    tot_sim = 1;
else
    tot_sim = AL_Strtucture.M/AL_Strtucture.h + 1;
end
%% 5th Parameter is Ground Truths
gt = varargin{5};
if ~numel(gt)
    error('No Hyperspectral Ground Truths');
end
%% Saving Test Samples Locations
TeC_Locations = cell(tot_sim, 1);
SVM_Pre_Class = cell(tot_sim, 1);
time = zeros(tot_sim, 1);
%% Start Active Learning Process For Several Classifiers
for iter = 1: tot_sim
    tic;
    fprintf('SVM Active Selection %d \n', iter)
    Tr = img(TrC(1,:), :);      %% Training Set
    Te = img(TeC(1,:), :);      %% Training Set
    TeC_Locations{iter} = TeC;
%% SVM Classifier
    SVM_Train = cell(numel(uc),1);
    SVM_W = zeros(size(Te, 1),numel(uc));
    for l = 1 : numel(uc)
        currentclass = (TrC(2,:) == uc(l));
        SVM_Train{l} = fitcsvm(Tr, currentclass, 'KernelFunction','polynomial',...
            'Standardize',true,'ClassNames',[false,true], 'KernelScale','auto');
        [~, Temp] = predict(SVM_Train{l}, Te);
        SVM_W(:,l) = Temp(:,2);
    end
    %% Compute the Output Class
    SVM_W = SVM_W';
    [~, SVM_Class_Results.map] = max(SVM_W);
    SVM_Pre_Class{iter} = SVM_Class_Results.map;
    %% Compute the Accuracy
    [SVM_Class_Results.OA(iter), SVM_Class_Results.kappa(iter),...
        SVM_Class_Results.AA(iter)] = My_Accuracy(TeC(2,:), ...
            SVM_Class_Results.map,(1:numel(uc)));
    %% Sample Selection 
            SVM_W = My_Membership(uc, SVM_W');
            SVM_Fuz = My_Fuzziness(SVM_W)';
            Pred = SVM_Class_Results.map;
            Pred  = [Pred; AL_Strtucture.Candidate_Set];
            Pred = [SVM_Fuz; Pred]';
            [A, ind] = sortrows(Pred, -1);
            [idx, ~] = find(A(:,4) ~= A(:,2));
            index_SVM_minME = ind(idx);
            if length(index_SVM_minME)>(AL_Strtucture.h)
                xp = index_SVM_minME(1 : AL_Strtucture.h)';
            else
                ind(idx) = [];
                index_SVM_minME = [index_SVM_minME' ind'];
                xp = index_SVM_minME(1 : AL_Strtucture.h)';
            end
            TrCNew = AL_Strtucture.Candidate_Set(:,xp);
            TrC = [TrC, TrCNew];
            AL_Strtucture.Candidate_Set(:,xp) = [];
            TeC = AL_Strtucture.Candidate_Set;
    time(iter,:) = toc; clear tic toc
%% End for Iter on AL
end
end
%% KNN Classifier
function [time, KNN_Class_Results] = My_KNN(varargin)
%% Compiling the Input
%% 1st Parameter is Hyperspectral Dataset
img = varargin{1};
if ~numel(img),error('No Hyperspectral Data');end
%% 2nd Parameter is Training Labels with Spatial Information
TrC = varargin{2};
if isempty(TrC)
    error('Please Include the Training Labels');
end
uc = unique(TrC(2,:));
%% 3rd Parameter is Test Labels with Spatial Information
if nargin > 2
    TeC = varargin{3};
else
    fprintf('Test Labeles Not Available, Thus we used Training as Validation');
    TeC = TrC;
end
%% 4th Parameter is Active Learning strtucture
if nargin > 3
    AL_Strtucture = varargin{4};
else
    AL_Strtucture = [];
end
if isempty(AL_Strtucture)
    tot_sim = 1;
else
    tot_sim = AL_Strtucture.M/AL_Strtucture.h + 1;
end
%% 4th Parameter is Ground Truths
if nargin > 4
    NN = varargin{5};
else
    NN = 20;
end
%% 5th Parameter is Ground Truths
gt = varargin{5};
if ~numel(gt)
    error('No Hyperspectral Ground Truths');
end
%% Saving Test Samples Locations
TeC_Locations = cell(tot_sim, 1);
time = zeros(tot_sim, 1);
%% Start Active Learning Process For Several Classifiers
for iter = 1: tot_sim
    tic;
    fprintf('KNN Active Selection %d \n', iter)
    Tr = img(TrC(1,:), :);      %% Training Set
    Te = img(TeC(1,:), :);      %% Training Set
    TeC_Locations{iter} = TeC;
%% KNN Training and Test
    OA = zeros(NN,1); kappa = zeros(NN,1); AA = zeros(NN,1); 
    KNN_W = cell(NN, 1); classp = cell(NN, 1);
    for ll = 1:NN
        Train = cell(numel(uc),1);
        score = zeros(size(Te,1),numel(uc));
        for l = 1 : numel(uc)
            currentclass = (TrC(2,:) == uc(l));
            Train{l} = fitcknn(Tr, currentclass, 'Standardize',true,'Distance','seuclidean',...
                            'ClassNames',[false,true],'NumNeighbors', ll);
            [~, Temp] = predict(Train{l}, Te);
            score(:,l) = Temp(:,2);
        end
%% Compute Accuracy
        [~, Cls] = max(score, [], 2);
        KNN_W{ll} = score';
        classp{ll} = Cls';
        [OA(ll), kappa(ll), AA(ll)] = My_Accuracy(TeC(2,:), ...
            classp{ll}, (1:numel(uc)));
    end
    [~, ind] = max(kappa(2:end));
    KNN_W = KNN_W{ind+1};
    KNN_Class_Results.map = classp{ind+1};
    KNN_Class_Results.OA(iter) = OA(ind+1);
    KNN_Class_Results.kappa(iter) = kappa(ind+1);
    KNN_Class_Results.AA(iter) = AA(ind+1);
    %% Plot the Accuracy According to the number of NN
%     figure(1);
%     set(gcf,'color','w');
%     hold on
%     plot(OA, 'sb', 'LineWidth', 2.5, 'markersize', 10, 'Linestyle', '-');
%     hold on
%     plot(kappa, '*r', 'LineWidth', 2.5, 'markersize', 10, 'Linestyle', '--');
%     hold on
%     plot(AA, 'ok', 'LineWidth', 2.5, 'markersize', 10, 'Linestyle', ':');
%     legend({'OA', 'kappa', 'AA'},'FontSize',12,...
%     'FontWeight','bold','Location','Best'); 
%     legend('boxoff'); grid on;
%     ylabel('Accuracy', 'FontSize',12,'FontWeight','bold'); 
%     xlabel('Nearest Neighbor', 'FontSize',12,'FontWeight','bold');
%     ax = gca;
%     outerpos = ax.OuterPosition;
%     ti = ax.TightInset; 
%     left = outerpos(1) + ti(1);
%     bottom = outerpos(2) + ti(2);
%     ax_width = outerpos(3) - ti(1) - ti(3);
%     ax_height = outerpos(4) - ti(2) - ti(4);
%     ax.Position = [left bottom ax_width ax_height];
%     saveas(figure(1), sprintf('NN%d.png', iter));
%     pause(1)
%     close all;
    %% Sample Selection
    KNN_W = My_Membership(uc, KNN_W');
    KNN_Fuz = My_Fuzziness(KNN_W)';
    Pred = KNN_Class_Results.map;
    Pred  = [Pred; AL_Strtucture.Candidate_Set];
    Pred = [KNN_Fuz; Pred]';
    [A, ind] = sortrows(Pred, -1);
    [idx, ~] = find(A(:,4) ~= A(:,2));
    index_KNN_minME = ind(idx);
    if length(index_KNN_minME)>(AL_Strtucture.h)
        xp = index_KNN_minME(1 : AL_Strtucture.h)';
    else
        ind(idx) = [];
        index_KNN_minME = [index_KNN_minME' ind'];
        xp = index_KNN_minME(1 : AL_Strtucture.h)';
    end
    TrCNew = AL_Strtucture.Candidate_Set(:,xp);
    TrC = [TrC, TrCNew];
    AL_Strtucture.Candidate_Set(:,xp) = [];
    TeC = AL_Strtucture.Candidate_Set;
    time(iter,:) = toc; clear tic toc
%% End for Iter on AL
end
end
%% ELM Classifier
function [time, ELM_Class_Results] = My_ELM(varargin)
%% Compiling the Input
%% 1st Parameter is Hyperspectral Dataset
img = varargin{1};
if ~numel(img),error('No Hyperspectral Data');end
%% 2nd Parameter is Training Labels with Spatial Information
TrC = varargin{2};
if isempty(TrC)
    error('Please Include the Training Labels');
end
uc = unique(TrC(2,:));
%% 3rd Parameter is Test Labels with Spatial Information
if nargin > 2
    TeC = varargin{3};
else
    fprintf('Test Labeles Not Available, Thus we used Training as Validation');
    TeC = TrC;
end
%% 4th Parameter is Active Learning strtucture
if nargin > 3
    AL_Strtucture = varargin{4};
else
    AL_Strtucture = [];
end
if isempty(AL_Strtucture)
    tot_sim = 1;
else
    tot_sim = AL_Strtucture.M/AL_Strtucture.h +1;
end
%% 6th Parameter is about the Fuzziness (High/Low)
if nargin > 4
    HN = varargin{5};
else
    HN = 20;
end
%% 7th Parameter is about the Fuzziness (High/Low)
if nargin > 5
    Act_Fun = varargin{6};
else
    Act_Fun = 'sin';
end
%% Saving Test Samples Locations
TeC_Locations = cell(tot_sim, 1);
ELM_Pre_Class = cell(tot_sim, 1);
time = zeros(tot_sim, 1);
%% Start Active Learning Process For Several Classifiers
for iter = 1: tot_sim
    tic;
    fprintf('ELM Active Selection %d \n', iter)
    Tr = img(TrC(1,:), :);      %% Training Set
    Te = img(TeC(1,:), :);      %% Training Set
    TeC_Locations{iter} = TeC;
%% ELM Classifiers Initialization for Binary Class
    Te_Mem = cell(1,HN);
    Te_Acc = zeros(1, HN);
    Tr_Acc = zeros(1, HN);
    for ll = 1: HN
        [~, Te_Mem{ll}, Tr_Acc(ll), Te_Acc(ll)] = ELM(Tr, TrC(2,:)',...
            Te, TeC', ll, Act_Fun);
        if round(Te_Acc(ll),2) > 99
            break;
        end
    end
    [~, ind1] = max(Te_Acc);
    %% Compute the Output Class
    MS = Te_Mem(ind1);
    ELM_W = cell2mat(MS);
    ELM_W = ELM_W';
    [~, ELM_Class_Results.map] = max(ELM_W);
    ELM_Pre_Class{iter} = ELM_Class_Results.map;
%% Compute the Accuracy
    [ELM_Class_Results.OA(iter), ELM_Class_Results.kappa(iter), ...
        ELM_Class_Results.AA(iter)] = My_Accuracy(TeC(2,:), ...
            ELM_Class_Results.map,(1:numel(uc)));
    %% Plot the Accuracy According to the number of HN's
%     figure(1);
%     set(gcf,'color','w');
%     hold on
%     plot(Tr_Acc, 'b', 'LineWidth', 2.5, 'markersize', 10, 'Linestyle', '-');
%     hold on
%     plot(Te_Acc, 'r', 'LineWidth', 2.5, 'markersize', 10, 'Linestyle', '--');
%     legend({'Training', 'Test'},'FontSize',18,...
%     'FontWeight','bold','Location','Best'); 
%     legend('boxoff'); grid on;
%     ylabel('Average Accuracy', 'FontSize',12,'FontWeight','bold'); 
%     xlabel('Number of Hidden Neurons', 'FontSize',12,'FontWeight','bold');
%     ax = gca;
%     outerpos = ax.OuterPosition;
%     ti = ax.TightInset; 
%     left = outerpos(1) + ti(1);
%     bottom = outerpos(2) + ti(2);
%     ax_width = outerpos(3) - ti(1) - ti(3);
%     ax_height = outerpos(4) - ti(2) - ti(4);
%     ax.Position = [left bottom ax_width ax_height];
%     saveas(figure(1), sprintf('HN%d.png', iter));
%     pause(1)
%     close all;
    %% Sample Selection
    ELM_W = My_Membership(uc, ELM_W');
    ELM_Fuz = My_Fuzziness(ELM_W)';
    Pred = ELM_Class_Results.map;
    Pred  = [Pred; AL_Strtucture.Candidate_Set];
    Pred = [ELM_Fuz; Pred]';
    [A, ind] = sortrows(Pred, -1);
    [idx, ~] = find(A(:,4) ~= A(:,2));
    index_ELM_minME = ind(idx);
    if length(index_ELM_minME)>(AL_Strtucture.h)
        xp = index_ELM_minME(1 : AL_Strtucture.h)';
    else
        ind(idx) = [];
        index_ELM_minME = [index_ELM_minME' ind'];
        xp = index_ELM_minME(1 : AL_Strtucture.h)';
    end
    TrCNew = AL_Strtucture.Candidate_Set(:,xp);
    TrC = [TrC, TrCNew];
    AL_Strtucture.Candidate_Set(:,xp) = [];
    TeC = AL_Strtucture.Candidate_Set;
    time(iter,:) = toc; clear tic toc
%% End for Iter on AL
end
end
%% ELM
function [Tr_Mem, Te_Mem, Tr_Acc, Te_Acc] = ELM(Tr, TrC, Te, TeC, hidden_node, Activ_Fun)
%% Training Data Prepration
train_data = [Tr TrC];
%% Test Data Prepration
TeC = TeC(:, 2);
test_data = [Te TeC];
%% Get the Unique Labels except 0
uc = unique(test_data(:,end));  %% Unique Labels
%% Preprocess to Generate the Membersihp Matrix
input_node = size(train_data(1:size(train_data,2)-1),2); 
output_node = length(unique(train_data(:,size(train_data,2))));
input_weigth = rand(input_node,hidden_node);
hidden_baise = rand(hidden_node,1); 
train_unary_class = zeros(size(train_data,1),output_node);  %% Unary Matrix
for r = 1:size(train_data,1)
    for c = 1:length(uc)
        if train_data(r,size(train_data,2)) == uc(c)
            train_unary_class(r,c) = 1;
        end
    end
end
test_unary_class = zeros(size(test_data,1),output_node);  %% Unary Matrix
for r = 1:size(test_data,1)
    for c = 1:length(uc)
        if test_data(r,size(test_data,2)) == uc(c)
            test_unary_class(r,c) = 1;
        end
    end
end
%% ELM: Step 2: Calculate the hidden layer output matrix
tr_data = train_data(:,1:end-1);
train_H = (tr_data*input_weigth)';
bias_matrix = repmat(hidden_baise,1, size(train_data,1));
H = bias_matrix + train_H;
 %% Activation Function %% Calculate hidden neuron output matrix Tr_out_H
    switch lower(Activ_Fun)
        case {'sig','sigmoid'}
            %%%%%%%% Sigmoid 
            Tr_out_H = 1 ./ (1 + exp(-H));
        case {'sin','sine'}
            %%%%%%%% Sine
            Tr_out_H = sin(H);
        case {'hardlim'}
            %%%%%%%% Hard Limit
            Tr_out_H = double(hardlim(H));
        case {'tribas'}
            %%%%%%%% Triangular basis function
            Tr_out_H = tribas(H);
        case {'radbas'}
            %%%%%%%% Radial basis function
            Tr_out_H = radbas(H);
            %%%%%%%% More activation functions can be added here                
    end
W = pinv(Tr_out_H)'*train_unary_class;   %% Beta matrix (output_weigths)
%% Training Accuracy
Tr_Mem = Tr_out_H'*W;  %%Traing Accuracy
%% Training Average 
m = 0;
for i = 1:size(train_unary_class,1)
        [~, ai]= max(train_unary_class(i,:));
        [~, bi]= max(Tr_Mem(i,:));
        if (ai~=bi)
            m = m+1;
        end
end
Tr_Acc = 100*(1-m/size(train_unary_class,1));
%% Testing Accuracy
tst_data = test_data(:,1:end-1); 
tst_H = (tst_data*input_weigth)';
bias_matrix = repmat(hidden_baise,1, size(tst_data,1));
tstH = bias_matrix + tst_H;
 %% Activation Function %% Calculate hidden neuron output matrix Tr_out_H
    switch lower(Activ_Fun)
        case {'sig','sigmoid'}
            %%%%%%%% Sigmoid 
            tst_out_H = 1 ./ (1 + exp(-tstH));
        case {'sin','sine'}
            %%%%%%%% Sine
            tst_out_H = sin(tstH);
        case {'hardlim'}
            %%%%%%%% Hard Limit
            tst_out_H = double(hardlim(tstH));
        case {'tribas'}
            %%%%%%%% Triangular basis function
            tst_out_H = tribas(tstH);
        case {'radbas'}
            %%%%%%%% Radial basis function
            tst_out_H = radbas(tstH);
            %%%%%%%% More activation functions can be added here                
    end
Te_Mem = tst_out_H'*W;
%% Testing Average
m = 0;
for i = 1:size(test_unary_class,1)
        [~, ai] = max(test_unary_class(i,:));
        [~, bi] = max(Te_Mem(i,:));   %% Te_Mem
        if (ai~=bi)
            m = m+1;
        end
end
Te_Acc = 100*(1-m/size(test_unary_class,1));
end
%% Membership
function SVM_score = My_Membership(uc, SVM_score)
%% Reformulate the Membership Matrix as per the defination
for r = 1:size(SVM_score,1)
    minVal = 0;
    maxVal = 1/numel(uc);
    [~, ind] = max(SVM_score(r,:));
    SVM_score(r,:) = minVal + (maxVal - minVal)*rand(1,numel(uc));
    AD1 = sum(SVM_score(r,:));
    AB1 = SVM_score(r,ind);
    AB2 = AD1 - AB1;
    AB3 = 1 - AB2;
    SVM_score(r,ind) = AB3;
end
end
%% Fuzziness
function MemberShip = My_Fuzziness(MemberShip)
%% Compute Fuzziness
Fuzziness = zeros(size(MemberShip,1),1);
for l = 1:size(Fuzziness,1)
    Fuzziness(l,:) = fuzziness(MemberShip(l,:));
end
Fuzziness = real(Fuzziness);
MemberShip = nonzeros(Fuzziness);
end
%% Fuzziness 
function E = fuzziness(u_j)
E = 0.01;
c = size(u_j,2);
flt_min = 1.175494e-38;
for i=1:c
    E = E-1/c*(u_j(1,i)*log2(u_j(1,i)+flt_min)+(1-u_j(1,i))*log2(1-u_j(1,i)+flt_min));
end
end
%% Accuracy
function [OA, kappa, AA] = My_Accuracy(True, Predicted, uc)
nrPixelsPerClass = zeros(1, length(uc))';
Conf = zeros(length(uc), length(uc));
for l_true = 1 : length(uc)
    tmp_true = find (True == (l_true));
    nrPixelsPerClass(l_true) = length(tmp_true);
    for l_seg = 1 : length(uc)
        tmp_seg = find (Predicted == (l_seg));
        nrPixels = length(intersect(tmp_true, tmp_seg));
        Conf(l_true, l_seg) = nrPixels;  
    end
end
diagVector = diag(Conf);
PC = (diagVector./(nrPixelsPerClass));
AA = mean(PC);
OA = sum(Predicted == True)/length(True);
kappa = (sum(Conf(:))*sum(diag(Conf)) - sum(Conf)*sum(Conf,2))...
    /(sum(Conf(:))^2 -  sum(Conf)*sum(Conf,2));
end
%% Good Luck!
