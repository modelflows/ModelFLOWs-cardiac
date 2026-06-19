clc; clear; close all;

%% PARAMETERS
% Number of components (mode 1)
ncomp = 3;

% Spatial and temporal dimensions
nx = 20;   % mode 2
ny = 24;   % mode 3
nz = 21;   % mode 4
nt = 10;   % mode 5

%% BUILD FULL 5D TENSOR
% Tensor has size: (ncomp, nx, ny, nz, nt)
Tensor = zeros(ncomp, nx, ny, nz, nt);

% Fill Tensor with a smooth test function depending on all indices
% You can replace this with your own data.
for c = 1:ncomp        % mode 1
    for i = 1:nx       % mode 2
        for j = 1:ny   % mode 3
            for k = 1:nz  % mode 4
                for t = 1:nt  % mode 5
                    x   = i / nx;
                    y   = j / ny;
                    z   = k / nz;
                    tau = t / nt;
                    Tensor(c,i,j,k,t) = ...
                        c * ( 2*x.^2 .* cos(2*(x*y+1)*(z-0.5)) ...
                        + sin(3*(x+2)*(y-1)) ...
                        - x.^3.*z - x.^2.*y.^2 ) ...
                        + 0.1 * sin(4*(x+2)*(tau+0.3));
                end
            end
        end
    end
end

%% COMPUTE SINGULAR VALUE DECAY FOR EACH MODE
P = ndims(Tensor);       % number of modes -> should be 5
sing_vals = cell(1,P);   % to store singular values per mode

for mode = 1:P
    % Mode-n unfolding of the tensor
    A = ndim_unfold(Tensor, mode);   % size: dim(mode) x (product of others)
    
    % Full (economy) SVD
    [~, S_mode, ~] = svd(A, 'econ');
    
    % Store singular values (diagonal of S)
    sing_vals{mode} = diag(S_mode);
end

%% CHOOSE TUCKER RANKS (ONE PER MODE)
% You can tune these based on the singular value decay.
n = [3, 6, 6, 6, 4];   % example ranks for modes 1..5

%% STANDARD HOSVD (USING FULL TENSOR)
[TT, S_core, U, sv, n_out] = hosvd(Tensor, n);

%% PLOT SINGULAR VALUE DECAY FOR EACH MODE
figure;
hold on;
for mode = 1:P
    s = sing_vals{mode};
    semilogy(1:length(s), s, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Mode %d', mode));
end
hold off;
xlabel('Singular value index');
ylabel('Singular value (log scale)');
title('Singular value decay in each mode (HOSVD)');
legend('show', 'Location', 'best');
grid on;