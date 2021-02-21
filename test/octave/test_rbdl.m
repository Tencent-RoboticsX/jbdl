addpath(genpath('mRBDL'))
addpath(genpath('ipPhysicalParams'))
addpath(genpath('ipTools'))
addpath(genpath('ipVideos'))

global ip;
global model;

ipParmsInit(0, 0, 0, 0);

model = model_create();

q = [0,  0.4765, 0, pi/6, pi/6, -pi/3, -pi/3];
PlotModel(model, q);
PlotCoMInertia(model, q);

% n = 100;
% for i=1:n
%     q(:,i) = zeros(7,1);
%     q(5,i) = LinearSpline(-pi/2, pi/2, i/n);
%     q(6,i) = LinearSpline(-pi/2, pi/2, i/n);
% end
% 
% PlotAVI('./ipVideos/t1.avi', q, n, model);

