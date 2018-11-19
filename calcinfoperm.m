function Iperm = calcinfoperm(x, xb, y, yb, Nperm)
% Calcualte null permutations

Iperm = zeros(Nperm,1);
Ntrl = length(x);
for pi=1:Nperm
    idx = randperm(Ntrl);
    Iperm(pi) = calcinfo(x(idx), xb, y, yb);
end
