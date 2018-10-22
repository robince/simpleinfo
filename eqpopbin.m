function xout = eqpopbin(x, nb)
% eqpopbin(x,nb)
% Bin a sequence of continuous values (x) into nb discrete
% categories which are approximately equally occupied

sx = sort(x);
% sx = unique(x);
N = length(sx);

% determine the edges of the bins
numel_bin = floor(N/nb); %number of values-per-bin
r = N - (numel_bin*nb);  % Remainder

indx = 1:numel_bin:numel_bin*nb;
indx(1:r) = indx(1:r) + (0:(r-1));
indx(r+1:end) = indx(r+1:end) + r;

edges = zeros(nb+1,1);
edges(1:nb) = sx(indx);
edges(nb+1) = sx(end) + 1;

% bin the data
[~, xout] = histc(x, edges);
% 0 based labelling
xout = xout - 1;

