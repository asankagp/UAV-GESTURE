%% Demo to compute P-CNN
% Report bugs to guilhem.cheron@inria.fr
%
% 
% ENABLE GPU support (in my_build.m) and MATLAB Parallel Pool to speed up computation (parpool) 
clear all
addpath('SVM Classification');
addpath('libsvm/matlab');

if ~isdeployed
    addpath('brox_OF'); % Brox 2004 optical flow
end
matconvpath = 'matconvnet-1.0-beta11'; % MatConvNet
run([matconvpath '/my_build.m']); % compile: modify this file to enable GPU support (much faster)
run([matconvpath '/matlab/vl_setupnn.m']) ; % setup  

%% OpenPose readings
% //     {1,  "Nose"},
% //     {2,  "Neck"},
% //     {3,  "RShoulder"},
% //     {4,  "RElbow"},
% //     {5,  "RWrist"},
% //     {6,  "LShoulder"},
% //     {7,  "LElbow"},
% //     {8,  "LWrist"},
% //     {9,  "RHip"},
% //     {10, "RKnee"}, -- remove for upper-body 
% //     {11, "RAnkle"}, -- remove for upper-body 
% //     {12, "LHip"},
% //     {13, "LKnee"}, -- remove for upper-body 
% //     {14, "LAnkle"}, -- remove for upper-body 
% //     {15, "REye"},
% //     {16, "LEye"},
% //     {17, "REar"},
% //     {18, "LEar"},

%% P-CNN computation
% ----- PARAMETERS --------
param=[];
% param.lhandposition=13; % pose joints positions in the structure (JHMDB pose format)
% param.rhandposition=12;
% param.upbodypositions=[1 2 3 4 5 6 7 8 9 12 13];
% param.lside = 40 ; % length of part box side (also depends on the human scale)

% UAV-GESTURE
param.lhandposition=7; % pose joints positions in the structure (UAV-GESTURE pose format)
param.rhandposition=4;
param.upbodypositions=[1 2 3 4 5 6 7 8 9 12 15 16 17 18];
param.lside = 100 ; % length of part box side (also depends on the human scale)

param.savedir = 'p-cnn_features_split1'; % P-CNN results directory
param.impath = 'H:\Datasets\UAVGESTURE\images' ; % input images path (one folder per video)

param.imext = '.png' ; % input image extension type
param.jointpath = 'UAVGESTURE/joint_positions' ; % human pose (one folder per video in which there is a file called 'joint_positions.mat')

param.cachepath = 'cache_UAVGESTURE'; % cache folder path
param.net_app  = load('models/imagenet-vgg-f.mat') ; % appearance net path
param.net_flow = load('models/flow_net.mat') ; % flow net path
param.batchsize = 128 ; % size of CNN batches
param.use_gpu = false ; % use GPU or CPUs to run CNN?
param.nbthreads_netinput_loading = 20 ; % nb of threads used to load input images
param.compute_kernel = true ; % compute linear kernel and save it. If false, save raw features instead.


% get video names
video_names = dir(param.impath);
video_names={video_names.name};
video_names=video_names(~ismember(video_names,{'.','..'}));

if ~exist(param.cachepath,'dir'); mkdir(param.cachepath) ; end % create cache folder

% 1 - pre-compute OF images for all videos
compute_OF(video_names,param); % compute optical flow between adjacent frames

% 2 - extract part patches
extract_cnn_patches(video_names,param)

% 3 - extract CNN features for each patch and group per video
extract_cnn_features(video_names,param)

for SplitT = 1:3    
    param.savedir = sprintf('p-cnn_features_split%d',SplitT);     % 'p-cnn_features_split1';'p-cnn_features_split2';
    param.partids = [1 2 3];
    
    % 4 - compute final P-CNN features + kernels
    compute_pcnn_features(video_names,param,SplitT); % compute P-CNN for splits (1,2,3)
    
    % Classification Accuracy
    Acc = classification(param,SplitT);
    Acccuracy(SplitT) = Acc(1);
end

fprintf('Action Recognition Accuracy: Acc = %f', mean(Acccuracy)); 


