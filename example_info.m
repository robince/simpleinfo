% Examples for calculating mutual information (MI) for salience of objects
% in scenes

%% 
% statistical sampling unit is "object"
% 76*8 objects presented to each subject ~600 samples
Nsamp = 600;

% generate some dummy values to illustrate the calculation
salience = randi(2,Nsamp,1)-1; % 0,1 coded low vs high saliency of each object
fixation = randi(2,Nsamp,1)-1; % 0,1 coded measure of fixation (could be fixated in first second, fixated at all etc.)

[I p] = calcinfo(salience,2,fixation,2);
% I is effect size, p gives p-value (sort of approximate)

% you might want to "bias correct" the value for later things like
% comparing to CMI
% bias is a function of the number of bins and number of trials

bias = mmbiasinfo(2,2,Nsamp);
% you can just subtract it off
Ibc = I - bias;

% alternatively there are permutation tests for statistics. For CMI /
% modulation permutation might be necessary - for the direct the p-value
% you get from the function.

% target identity
target = randi(2,Nsamp,1)-1; % 0,1 coded target vs distractor

% conditional mutual info
%I I(salience; fixation | target).
% There are two ways to do this... the first is with calcmi

[CMI p] = calccmi(salience,2,fixation,2,target,2);
bias = mmbiascmi(2,2,2,Nsamp);
CMIbc = CMI - bias;

% now you can compute your interaction / modulation index
ModIdx = Ibc - CMIbc;

% alternatively, you can manually calculate the conditional info values
% which might give more information (ie you can say info is higher in one
% vs other condition).

individualcmi = zeros(1,2);
Ntrltarget = zeros(1,2);
for i=1:2
    idx = target==i-1;
    individualcmi(i) = calcinfo(salience(idx),2,fixation(idx),2);
    Ntrltarget(i) = sum(idx);
end
% cmi is the avarage of these
Ptarget = Ntrltarget ./ Nsamp;
CMImanual = sum(Ptarget.*individualcmi);
% (should be the same CMI above, but can look at individualcmi)

% you can replace 'target' with some other property to condition out if you
% want. Also you can combine multiple things to condition out, but that
% might run into data limits so I woudl start with the above.


%%
%If you have instead a continuous fixation time then it is the same idea
% but you use the gcmi functions.

Nsamp = 600;

% generate some dummy values to illustrate the calculation
salience = randi(2,Nsamp,1)-1; % 0,1 coded low vs high saliency of each object
fixation = randn(Nsamp,1); % now normally distributed fixaction times (dont have to be normal)

% the gcmi method relies on a first transformation - the gcmi_ functions do
% this automatically but I do it explicitly here as if you do permutations
% you need to be aware of it to save doing it each time.
% samples (objects) must be the first axis
cfix = copnorm(fixation);
% I = mi_model_gd(cfix, salience, 2);
% there is an alternative which might be better for interactions
I = mi_mixture_gd(cfix, salience, 2);
% here it might be better to use mixture

% Here the conditional mutual information you need to calculate yourself
% like method 2 above.
% I think here it is better to copnorm within each target value separately.
individualcmi = zeros(1,2);
Ntrltarget = zeros(1,2);
for i=1:2
    idx = target==i-1;
    individualcmi(i) = mi_mixture_gd(copnorm(fixation(idx)), salience(idx), 2);
    Ntrltarget(i) = sum(idx);
end
% cmi is the avarage of these
Ptarget = Ntrltarget ./ Nsamp;
CMImanual = sum(Ptarget.*individualcmi);

% like before you can do the modulation index
I - CMImanual

% using mixture vs model function is a bit unsure because this is a
% different application than what I am used to.  Maybe try mixture first.

%%
% For pixel wise calculation (e.g. other paradigms).
Nx = 124;
Ny = 768;
salience = rand(Nx, Ny); % [0,1] continuous salience value for each pixel
fixation = randi(2, Nx, Ny) - 1; % 0,1 coded binary fixation result

% key point is here sample unit is pixels instead of objects
% otherwise everything works the same
% you need to flatten out the pixel samples and make them first axis

% now salience is the continuous variable, fixation is discrete
csalience = copnorm(salience(:));
I = mi_mixture_gd(csalience, fixation(:), 2)

% or could do the same with a continuous measure of fixation time
fixation = randn(Nx,Ny);

csalience = copnorm(salience(:));
cfix = copnorm(fixation(:));
I = mi_gg(cfix, csalience);

% more conditional possible as required....

