function bias = mmbiascmi(Nx, Ny, Nz, Ntrl)
% bias = mmbias(Nx, Ny, Ntrl)
% Miller-Madow bias estimate for subtraction from uncorrected binned
% information values
% Nx - number of bins for first variable
% Ny - number of bins for second variable
% Ntrl - number of trials

if ~isscalar(Nx) || ~isscalar(Ny) || ~isscalar(Nz) || ~isscalar(Ntrl)
    error('mmbias: only scalar arguments supported')
end
bias = Nz.*(Nx-1).*(Ny-1) ./ (2.*Ntrl*log(2));
