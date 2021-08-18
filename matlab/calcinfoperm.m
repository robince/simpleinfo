function Iperm = calcinfoperm(x, xb, y, yb, Nperm, bias, beta)
% Iperm = calcinfo(x, xb, y, yb, beta)
% Calculate samples from the permutation null that X and Y are 
% independent by shuffling the relationship

if nargin<6
    bias = true;
end
if nargin<7
    beta = 0;
end
Iperm = zeros(Nperm,1);
Ntrl = length(x);
for pi=1:Nperm
    idx = randperm(Ntrl);
    Iperm(pi) = calcinfo(x(idx), xb, y, yb, false, beta);
end

if bias
    Iperm = Iperm - mmbiasinfo(xb,yb,Ntrl);
end
