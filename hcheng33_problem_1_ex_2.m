%% Problem 1 Part A

N = 64;

% tuning curve
g = gausswin(N,5).*(cos(2*pi*(0:(N-1))/10).');

% sample stimuli
X = 2*rand(N,1);

% response distribution
P = @(X) poissrnd(exp(g'*X),100,1);

figure();
for i = 1:4
    R = P(X);

    subplot(2,2,i);
    histogram(R);
    title('Draw ',i)
end
sgtitle('Psrt A: Histogram of r Distribution');

%% Problem 1 Part B

M = 10000;

r = zeros(M,1);
X = 2*rand(M,N);
rate_all = X*g;

% draw M samples
for i = 1:M
    %x = X(i,:);
    rate = rate_all(i);
    r(i) = poissrnd(exp(rate),1,1);
end

% solve for r = Xg
g_hat_b = X\r;

% plot
figure();
plot(g); hold on; plot(g_hat_b);
title("Part B: Estimate g and Real g");
legend("Real g", "Estimate g");

%% Problem 1 Part C

% define different numbers of M
M_trials = [N, N*2, N/2];
g_hat_c = zeros(N,length(M_trials));

figure();
plot(g); hold on;

% deriving g_hat from cost function for different numbers of M
for i = 1:length(M_trials)
    g_hat_c(:,i) = problem_1_part_c(M_trials(i),N);
    plot(g_hat_c(:,i)); hold on;
end

title("Part C: Estimate g for differnet # of M");
legend('g','M = N','M = 2N', 'M = N/2');

%% Problem 1 Part D

M_trials = 1000;
sig = 0.1;
A = 2;

figure();
plot(g); hold on;

% deriving g_hat from cost function with gaussian prior
g_hat_d_gaussian = problem_1_part_d_gaussian(M_trials,N,sig,A);
plot(g_hat_d_gaussian); hold on;

% deriving g_hat from cost function with poisson prior
g_hat_d_poisson = problem_1_part_d_poisson(M_trials,N,sig,A);
plot(g_hat_d_poisson); hold on;

title("Part D: g Estimate with different priors")
legend('g real','g with gaussian prior','g with poisson prior');
%% Problem 1 Part E

M = 1000;
N = 64;
ind = 0;
A_all = [0.01, 1, 2, 10];
sig_all = [0.01, 0.1, 1];

% varying A and sig for gaussian prior
figure();
for i = 1:length(A_all)
    xx = A_all(i)*rand(M,N);
    for j = 1:length(sig_all)
        ind = ind+1;
        g_hat = problem_1_part_d_gaussian(M,N,sig_all(j),A_all(i));
        p = poissrnd(exp(g_hat'.*xx));
        subplot(4,3,ind);
        histogram(p);
    end
end
sgtitle("Gaussian Prior")

% varying A and sig for Poisson prior
ind = 0;
figure();
for i = 1:length(A_all)
    xx = A_all(i)*rand(M,N);
    for j = 1:length(sig_all)
        ind = ind+1;
        g_hat = problem_1_part_d_poisson(M,N,sig_all(j),A_all(i));
        p = poissrnd(exp(g_hat'.*xx));
        subplot(4,3,ind);
        histogram(p);
    end
end
sgtitle("Poisson Prior")


%% Functions

function g_hat = problem_1_part_c(M,N)
    g = gausswin(N,5).*(cos(2*pi*(0:(N-1))/10).');

    r = zeros(M,1);
    X = 2*rand(M,N);
    rate_all = X*g;
    
    % sample draw
    for i = 1:M
        rate = rate_all(i);
        r(i) = poissrnd(exp(rate),1,1);
    end

    % cost function definition
    cost_func = @(g) -sum((r.*(X*g)) - exp(X*g));
    g_hat = fminunc(cost_func,zeros(N,1));
end

function g_hat = problem_1_part_d_gaussian(M,N,sig,A)
    g = gausswin(N,5).*(cos(2*pi*(0:(N-1))/10).');

    r = zeros(M,1);
    X = A*rand(M,N);
    rate_all = X*g;
    
    % sample draw
    for i = 1:M
        rate = rate_all(i);
        r(i) = poissrnd(exp(rate),1,1);
    end

    % cost function definition
    cost_func = @(g) -(sum(-0.5*(r-X*g)'*((sig^2*eye(M)^-1)*(r-X*g))) - (0.5*g'*((sig^2*eye(N))^-1)*g));
    g_hat = fminunc(cost_func,zeros(N,1));
end

function g_hat = problem_1_part_d_poisson(M,N,sig,A)
    g = gausswin(N,5).*(cos(2*pi*(0:(N-1))/10).');

    r = zeros(M,1);
    X = A*rand(M,N);
    rate_all = X*g;
    
    % sample draw
    for i = 1:M
        rate = rate_all(i);
        r(i) = poissrnd(exp(rate),1,1);
    end

    % cost function defintion
    cost_func = @(g) -(sum((r.*(X*g)) - exp(X*g)) -0.5*(diff(g)'*((sig^2*eye(N-1))^-1)*diff(g)) );
    g_hat = fminunc(cost_func,zeros(N,1));
end