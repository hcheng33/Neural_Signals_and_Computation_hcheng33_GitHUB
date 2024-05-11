%%  load data
data = load("sample_dat.mat");

trials = length(data.dat);
psth = zeros(size(data.dat(1).spikes));

for i = 1:trials
    psth = psth + data.dat(i).spikes;
end

% plot(psth(1,:));
% neuron = psth(1,:);
%% Problem 2 Part A

t = (1:400)*1e-3;
psth_smooth = zeros(53,400);

% find smoothed psth by fitting Guassina process
% default kernel is squared exponential
for i = 1:length(psth(:,1))
    neuron = psth(i,:);
    mdl = fitrgp(t',neuron');
    psth_smooth(i,:) = resubPredict(mdl)';
end

%% Problem 2 Part B

% take pca of PSTH and smoothed PSTH
pca_raw = pca(psth);
pca_smooth = pca(psth_smooth);

figure();
subplot(1,2,1);
plot3(pca_raw(:,1),pca_raw(:,2),pca_raw(:,3));
title('First 3 PCA Components of PSTH')

subplot(1,2,2);
plot3(pca_smooth(:,1),pca_smooth(:,2),pca_smooth(:,3))
title('First 3 PCA Components of Smoothed PSTH')

%% Problem 2 Part C

% neural acitviet via GPFA
result = neuralTraj(0,data.dat);
[estParams, seqTrain] = postprocess(result);

% plot time-courses
plot3D(seqTrain,'xorth', 'dimsToPlot', 1:3);

%% Problem 2 Part D

% the variability of the neural signal among different trials might also be
% related to the specific movement of the test subject when it is reaching
% for the goal

% computationally we can correlate the actual movement behavior to the
% neural signals to see if there's any clear relationship between the two

%% functions

% I was originally trying to derive the cost function mathematically and
% fit the Gaussian Process

% Results were bad so I reverted to matlab built in function
function x_hat = gp_cost_func(psth,A,l,sig)
    K = zeros(400,400);
    t = (1:400) * (1e-3);

    for i = 1:400
        for j = 1:400
            K(i,j) = A*(exp(-((t(i)-t(j))^2)/l));
        end
    end

    cost_func = @(x) -((sum(-0.5*(psth-x)*((sig^2*eye(400)^-1)*(psth-x)'))) - (0.5*x*inv(K)*x'));
    x_hat = fmincon(cost_func,ones(1,400),[],[],[],[],zeros(1,400),[]);
end

