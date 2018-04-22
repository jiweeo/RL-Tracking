
addpath('test/');
addpath(genpath('utils/'));

load('models/net_rl.mat');

data_root = 'data/OTB100/';
file = fopen('filename');

pre = zeros(50,98);
filelist={};
for i=1:98
    filelist{i} = fgets(file);
end

tot_p = 0;
init_settings;
run(matconvnet_path);  
    
opts.visualize = false;
opts.printscreen = false;

for j=1:98
    vid_path=strcat(data_root,filelist{j}); 
    rng(1004);
    [results, t, p] = adnet_test(net, vid_path, opts);
    pre(:,j) = p;
    fprintf('[%d/98] time: %f\n', j, t);
end

save('res.mat','pre');