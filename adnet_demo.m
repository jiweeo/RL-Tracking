function [t,p, results] = adnet_demo (vid_path)
% ADNET_DEMO Demonstrate `action-decision network'
% 
% Sangdoo Yun, 2017.

if nargin < 1
    vid_path = 'data/OTB100/Bolt';
end

addpath('test/');
addpath(genpath('utils/'));

init_settings;

run(matconvnet_path);

load('models/net_rl.mat');

opts.visualize = false;
opts.printscreen = true;

rng(1004);
[results, t, p] = adnet_test(net, vid_path, opts);

fprintf('precision: %f, fps: %f\n', p(20), size(results, 1)/t);

