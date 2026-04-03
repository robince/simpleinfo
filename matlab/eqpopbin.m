function [xbin, edges] = eqpopbin(x, nb)
% eqpopbin(x,nb)
% Bin a sequence of continuous values (x) into nb discrete
% categories which are approximately equally occupied
% Outputs:
% xbin: x binned into integer values (values 0:nb-1)
% edges: nb-1 bin edges

sx = sort(x);
% sx = unique(x);
N = length(sx);

% determine the edges of the bins
numel_bin = floor(N/nb); %number of values-per-bin
r = N - (numel_bin*nb);  % Remainder
if numel_bin == 0
    error('eqpopbin: nb cannot exceed the number of samples');
end

indx = 1:numel_bin:numel_bin*nb;
indx(1:r) = indx(1:r) + (0:(r-1));
indx(r+1:end) = indx(r+1:end) + r;

edges = zeros(nb+1,1);
edges(1:nb) = sx(indx);
edges(nb+1) = sx(end) + 1;

% bin the data
[~, xbin] = histc(x, edges);
% 0 based labelling
xbin = xbin - 1;
