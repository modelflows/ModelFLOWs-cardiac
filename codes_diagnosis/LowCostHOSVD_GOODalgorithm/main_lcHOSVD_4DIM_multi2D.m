clc; clear; close all;

%% PARAMETERS
% Number of components in mode 1
ncomp = 3;

% Spatial and temporal dimensions
nx = 30;      % mode 2
ny = 36;      % mode 3
nt = 20;      % mode 4

%% BUILD FULL 4D TENSOR
% Tensor has size (ncomp, nx, ny, nt)
Tensor = zeros(ncomp, nx, ny, nt);

% Fill Tensor with a smooth test function depending on all indices
% You can replace this by your own data later.
for c = 1:ncomp         % mode 1
    for i = 1:nx        % mode 2
        for j = 1:ny    % mode 3
            for t = 1:nt % mode 4
                x   = i / nx;
                y   = j / ny;
                tau = t / nt;
                Tensor(c,i,j,t) = ...
                    c * ( 2*x.^2 .* cos(2*(x*y+1)*(y-0.5)) ...
                    + sin(3*(x+2)*(y-1)) ...
                    - x.^2.*y.^2 ) ...
                    + 0.1 * sin(4*(x+2)*(tau+0.3));
            end
        end
    end
end

%% DEFINE REDUCED INDICES IN MODES 2 AND 3
% We subsample:
% - every 2 points in mode 2 (nx)
% - every 3 points in mode 3 (ny)
IndexI = 1:2:nx;   % subsampling in mode 2
IndexJ = 1:3:ny;   % subsampling in mode 3

%% BUILD REDUCED TENSOR
% Reduced tensor keeps:
% - all components in mode 1
% - subsampled indices in modes 2 and 3
% - all time indices in mode 4
Tensor_reducido = Tensor(:, IndexI, IndexJ, :);

%% TUCKER RANKS (ONE PER MODE)
% Adjust these ranks depending on desired compression
n = [3, 6, 6, 4];   % [rank_mode1, rank_mode2, rank_mode3, rank_mode4]

%% LOW-COST HOSVD USING REDUCED TENSOR (lcSVD in modes 2 and 3)
% Tbig: full tensor
% Tred: reduced tensor
[TT, S, U, sv, n_out] = hosvd_lc_multi2D(Tensor, Tensor_reducido, n, IndexI, IndexJ);

%% CHECK RECONSTRUCTION ERROR
relerr = norm(Tensor(:) - TT(:)) / norm(Tensor(:));
fprintf('Relative reconstruction error: %.3e\n', relerr);