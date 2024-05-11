%% load data

data = load("exampleData.mat");
Data = data.Data;

%% Problem 3 Part A

% plot heatmap of Condition 27
data_27 = data.Data(27).A;

heatmap(data_27');
title("Heatmap of Condition 27")


%% Problem 3 Part A

% A_hat = part_b_cost_func(x);
% I was originally trying to derive A hat by fitting cost function we
% derived, but none of my computers were able to carry out the full scale
% of the calculations and thus I reverted to a simple linear regression

x = data_27;

% Estimating the A_hat matrix
y = x(2:end,:) - x(1:end-1,:);
AA = x(1:end-1,:)\y;

dt = 10e-3;
xx = zeros(61,218);

% Reconstructing Data from A_hat matrix
xx(2:end,:) = (inv(1-AA*dt) * (xx(1:end-1,:))')';

% Plotting Error Histrogram
err_raw = zeros(218,1);
for n = 1:218
    err_raw(n) = abs(norm((xx(:,n))-norm(x(:,n)))/norm(x(:,n)));
end

figure();
histogram(err_raw);
title('Histogram of Errors (Raw Data)')

% The signal reconstruction was really bad because of the A_hat matrix
% The inverse of the matrix is close to singular/ working precision

%% Problem 3 Part C

% Getting data across all trials
data_all = [];
for i = 1:108
    data_all = [data_all;Data(i).A];
end

% % Getting the first 6 PCA components
% pca_comp = pca(data_all','NumComponents',6);
% pca_comp = reshape(pca_comp,[108,61,6]);
% 
% % deriving A_hat matrix from cost function
% pca_cost_func = @(A) part_b_cost_func(pca_comp,A); 
% A_hat = fminunc(pca_cost_func,zeros(6,6));

% Getting the first 6 PCA components
[coeff, score_all, latent, ~, explained, mu] = pca(data_all,'NumComponents',6);
score = reshape(score_all,[61,108,6]);

% deriving A_hat matrix from cost function
pca_cost_func = @(A) part_b_cost_func(score,A); 
A_hat = fminunc(pca_cost_func,zeros(6,6));

% reconstructing signal from A_hat matrix and inverse PCA
xx_pca = (squeeze(score(:,27,:)))';
xx_pca(:,2:end) = dt*A_hat*xx_pca(:,1:end-1) + xx_pca(:,1:end-1);

pca_recon = xx_pca' * coeff' + repmat(mu,61,1);

% plotting the pca and reconstructed pca
figure();
for i = 1:6
    subplot(2,3,i);
    plot(score(:,27,i)); hold on; plot(xx_pca(i,:));
end
title("PCA components vs Reconstructed PCA components")

% plotting the histogrma of Errors
err_pca = zeros(218,1);
for n = 1:218
    err_pca(n) = abs(norm((pca_recon(:,n))-norm(data_27(:,n)))/norm(data_27(:,n)));
end

figure();
histogram(err_pca);
title('Histogram of Errors (PCA)')

%% Problem 3 Part D

% plot the first two dynamical principal dimensions
for i = 1:108
    plot(score(:,i,1),score(:,i,2)); hold on;
end
title("First Two Dynamical Principal Dimensions")

%% Problem 3 Part E

% jPCA
jPCA_params.softenNorm = 5;
jPCA_params.suppressBWrosettes = true;
jPCA_params.suppressHistograms = true;

times = -50:10:150;
jPCA_params.numPCs = 6;
[Projection, Summary] = jPCA(data.Data, times,jPCA_params);

phaseSpace(Projection, Summary);
printFigs(gcf,'.','-dpdf','Basic jPCA plot');

% eigen spectra
A_jpca = Summary.jPCs;

A_jpca_eig = eig(A_jpca);
A_hat_eig = eig(A_hat);

% the eigenvalues of the jPCA contain imaginary parts as opposed to the PCA
% only having real parts because it is a revolution about the origin

%% function
function val = part_b_cost_func(xx,A)
    dt = 10e-3;
    sig = 0.1;
    val = 0;

    for i = 1:108
        x = squeeze(xx(:,i,:));
        val = val + sum(0.5*(x(2:end,:)- (x(1:end-1,:)*(A*dt) + x(1:end-1,:))) * ...
            ((sig^2*eye(6)*dt)^(-1)) * (x(2:end,:)-(x(1:end-1,:)*(A*dt) + x(1:end-1,:)))','all');
    end

end

