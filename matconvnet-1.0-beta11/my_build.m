function my_build
% script to build matconvnet-1.0-beta11

matconvnetRoot = fileparts(mfilename('fullpath'));
run(fullfile(matconvnetRoot, 'matlab', 'vl_setupnn.m') );

%% CPU
vl_compilenn('enableImreadJpeg', true)

%% GPU support + other options (replace the corresponding paths)
% vl_compilenn('enableGpu', true, ...
%     'cudaRoot', '/meleze/data0/libs/cuda-6.5', ...
%     'cudaMethod', 'nvcc', ...
%     'enableCudnn', true, ...
%     'cudnnRoot', '/sequoia/data1/aosokin/software/cudnn-6.5-linux-x64-v2',  ...
%     'enableImreadJpeg', true, ...
%     'verbose', false ); 


%% GPU support + other options (replace the corresponding paths)
% vl_compilenn('enableGpu', true, ...
%     'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2', ...
%     'cudaMethod', 'nvcc', ...
%     'enableImreadJpeg', true, ...
%     'verbose', false ); 
