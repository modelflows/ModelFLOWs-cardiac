clc; clear; close all;

%% PARAMETERS
% Number of components in mode 1
ncomp = 3;

% Spatial/temporal dimensions
nx = 20;      % mode 2
ny = 24;      % mode 3
nz = 21;      % mode 4
nt = 10;      % mode 5

%% BUILD FULL 5D TENSOR
% Tensor has size (ncomp, nx, ny, nz, nt)
Tensor = zeros(ncomp, nx, ny, nz, nt);

% Fill Tensor with a smooth function of the indices
% You can replace this by your own data later.
for c = 1:ncomp
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                for t = 1:nt
                    x = i / nx;
                    y = j / ny;
                    z = k / nz;
                    tau = t / nt;
                    % Example field depending on all coordinates + component
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

%% DEFINE REDUCED INDICES IN MODES 2, 3, 4
% We subsample:
% - every 2 points in mode 2 (nx)
% - every 4 points in mode 3 (ny)
% - every 3 points in mode 4 (nz)

IndexI = 1:2:nx;   % subsampling in mode 2 (nx)
IndexJ = 1:4:ny;   % subsampling in mode 3 (ny)
IndexK = 1:3:nz;   % subsampling in mode 4 (nz)

%% BUILD REDUCED TENSOR
% Reduced tensor keeps:
% - all components in mode 1
% - subsampled indices in modes 2, 3, 4
% - all time indices in mode 5
Tensor_reducido = Tensor(:, IndexI, IndexJ, IndexK, :);

%% TUCKER RANKS (ONE PER MODE)
% You can adjust these depending on how much compression you want
n = [3, 5, 5, 5, 3];   % [rank_mode1, rank_mode2, ..., rank_mode5]

%% LOW-COST HOSVD USING REDUCED TENSOR
% Tbig: full tensor
% Tred: reduced tensor
[TT, S, U, sv, n_out] = hosvd_lc_multi3D(Tensor, Tensor_reducido, n, IndexI, IndexJ, IndexK);

%% CHECK RECONSTRUCTION ERROR
relerr = norm(Tensor(:) - TT(:)) / norm(Tensor(:));
fprintf('Relative reconstruction error: %.3e\n', relerr);