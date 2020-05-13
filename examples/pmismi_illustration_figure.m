addpath('../matlab')

% number of sampled points for calculation
% Ncalc = 10000000; % used for figure
Ncalc = 100000; % quicker

Ncol = 5; % number of examples
Nrow = 4;

figure
if exist('cbrewer','file')
    % cbrewer strongly recommended: 
    % https://uk.mathworks.com/matlabcentral/fileexchange/34087-cbrewer-colorbrewer-schemes-for-matlab
    cm = flipud(cbrewer('div','RdBu',256));
else
    % matlab doesn't have a builtin divergent colormap :(
    cm = parula;
end
colormap(cm)

% whether to show a colorbar on each figure
cbar = false;
% cbar = true;

% Gaussian Data
c = 0.8;
Cxy = [1 c; c 1];
xy = mvnrnd([0 0], Cxy, Ncalc);
x = xy(:,1);
y = xy(:,2);
xrange = [-4 4];
yrange = xrange;
plot_data(x,y,xrange,yrange,Nrow,Ncol,1,cbar);

% Gaussian Data
c = -0.8;
Cxy = [1 c; c 1];
xy = mvnrnd([0 0], Cxy, Ncalc);
x = xy(:,1);
y = xy(:,2);
xrange = [-4 4];
yrange = xrange;
plot_data(x,y,xrange,yrange,Nrow,Ncol,2,cbar)

% Gaussian Data
c = 0.4;
Cxy = [1 c; c 1];
xy = mvnrnd([0 0], Cxy, Ncalc);
x = xy(:,1);
y = xy(:,2);
xrange = [-4 4];
yrange = xrange;
plot_data(x,y,xrange,yrange,Nrow,Ncol,3,cbar)

% Smile
x = linspace(-1,1,Ncalc)';
r = (rand(Ncalc,1)*2)-1;
y = 2*(x.^2) + r;
xrange = [-1 1];
yrange = [-1 3];
plot_data(x,y,xrange,yrange,Nrow,Ncol,4,cbar)

% Circle
x = linspace(-1,1,Ncalc)';
r = randn(Ncalc,1)./8;
y = cos(x.*pi) + r;
r = randn(Ncalc,1)./8;
x = sin(x.*pi) + r;

xrange = [-1.5 1.5];
yrange = xrange;
plot_data(x,y,xrange,yrange,Nrow,Ncol,5,cbar)

% set(gcf, 'Pos', [ 214         427        1200         699])

function plot_data(x,y,xrange,yrange,Nrow,Ncol,subi,cbar)
% plot a stack of 4 figures:
% data and bin edges
% joint probability dist
% PMI 
% data colored by SMI

% subsampled number of points to plot in scatter
Nplot = 1000;
alpha = 0.4;

Ntrl = size(x,1);
idx = randperm(Ntrl,Nplot);
% bin data
Nbin = 16;
[bx, xedge] = eqpopbin(x,Nbin);
[by, yedge] = eqpopbin(y,Nbin);
xedge = xedge(2:end-1);
yedge = yedge(2:end-1);

ax = [];
% raw data
ax(1) = subplot(Nrow, Ncol, subi);
scatter(x(idx),y(idx),'filled','MarkerFaceAlpha',alpha)
if cbar
    cb = colorbar;
    set(cb,'Visible','off')
end
hold on
for ei=1:Nbin-1
    yline(yedge(ei),'k');
    xline(xedge(ei),'k');
end

% PMI values
ax(3) = subplot(Nrow, Ncol, subi+2*Ncol);
[I, PMI] = calcpmi(bx, Nbin, by, Nbin,[],0.5);
[xax, yax, img] = bin_img_plot(PMI, xedge, xrange, yedge, yrange);
imagesc(xax,yax,img)
caxis(ax(3),[-1 1]*max(abs(caxis)))
if cbar
    colorbar
end
I = I - mmbiasinfo(Nbin,Nbin,Ntrl);
title(ax(1), sprintf('%.2f bits',I));

% SMI values
ax(4) = subplot(Nrow, Ncol, subi+3*Ncol);
[~, SMI] = calcsmi(bx,Nbin,by,Nbin,0.5);
scatter(x(idx),y(idx),[],SMI(idx),'filled','MarkerFaceAlpha',alpha)
caxis(ax(4),[-1 1]*max(abs(caxis)))
if cbar
    colorbar
end

% joint binned probability distribution
ax(2) = subplot(Nrow, Ncol, subi+Ncol);
% (C + beta) / (N + (beta*C.size))
beta = 0.5; % KT estimate
Pxy = (accumarray([bx+1 by+1],1)+beta)./(Ntrl+beta*Nbin*Nbin);
[xax, yax, img] = bin_img_plot(Pxy, xedge, xrange, yedge, yrange);
imagesc(xax,yax,img)
caxis(ax(2),[-1 1]*max(abs(caxis)))
% for some reason setting cb limits only works if this is the last plot in
% the function
if cbar
    cb = colorbar;
    cb.Limits = [0 max(abs(caxis))];
end

axis(ax,'square')
set(ax,'XLim',xrange,'YLim',yrange,'YDir','normal')
set(ax,'XTick',[],'YTick',[],'XColor','none','YColor','none')

end

function [x, y, img] = bin_img_plot(dat, xedges, xlim, yedges, ylim)
% plot values for each joint bin as an image in original data space
Ngrid = 100;
x = linspace(xlim(1),xlim(2),Ngrid);
y = linspace(ylim(1),ylim(2),Ngrid);
[X, Y] = meshgrid(x,y);
img = NaN(size(X));

Nbin = size(dat,1);
for xi=1:Nbin
    if xi==1
        xidx = X<xedges(1);
    elseif xi==Nbin
        xidx = X>xedges(Nbin-1);
    else
        xidx = xedges(xi-1)<X & X<xedges(xi);
    end
    for yi=1:Nbin
        if yi==1
            yidx = Y<yedges(1);
        elseif yi==Nbin
            yidx = Y>yedges(Nbin-1);
        else
            yidx = yedges(yi-1)<Y & Y<yedges(yi);
        end
        img(xidx & yidx) = dat(xi,yi);
    end
end
end

