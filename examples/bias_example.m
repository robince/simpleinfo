
Ntrl = 500;

Nrep = 1000;

I = zeros(Nrep,1);
for ri=1:Nrep
    x = randi(2,Ntrl,1)-1;
    y = x; y(randperm(Ntrl,round(Ntrl/2))) = randi(2,round(Ntrl/2),1)-1;
    I(ri) = calcinfo(x,2,y,2);
end

%%

Ntrl = 100;

Nrep = 1000;

I = zeros(Nrep,1);
for ri=1:Nrep
    x = randi(2,Ntrl,1)-1;
    y = randi(2,Ntrl,1)-1;
    I(ri) = calcinfo(x,2,y,2,false) - mmbiasinfo(2,2,Ntrl);
end
