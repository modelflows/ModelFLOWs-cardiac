clc; clear; close all;

nx = 100;
ny = 102;
nz = 104;

Tensor = zeros(nx,ny,nz);
for i = 1:nx
    for j = 1:ny
        for k = 1:nz
            Tensor(i,j,k) = i + j - k;
        end
    end
end

IndexK = 1:4:nz;                 % submuestreo en la 3ª dimensión
Tensor_reducido = Tensor(:,:,IndexK);

n = [2 2 2];                     % rangos Tucker

[TT, S, U, sv, n_out] = hosvd_lc_modo3(Tensor, Tensor_reducido, n, IndexK);

% Compara con el tensor original
relerr = norm(Tensor(:) - TT(:)) / norm(Tensor(:))