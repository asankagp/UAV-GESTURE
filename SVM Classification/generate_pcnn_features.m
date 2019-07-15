%%compute the pcnn features
clear
clc
close all

param = [];
param.lside = 40 ; % length of part box side (also depends on the human scale)
param.savedir = 'p-cnn_features_2patch_split1'; % P-CNN results directory
param.cachepath = 'cache'; % cache folder path
param.use_gpu = true ; % use GPU or CPUs to run CNN?
param.compute_kernel = true ; % compute linear kernel and save it. If false, save raw features instead.
param.do_dyn = 1 ; % use dynamic features (differences)
param.do_acc = 0 ; % use differences of dynamic features
param.do_max = 1 ; % use max aggregation
param.do_min = 1 ; % use min aggregation
param.do_std = 0 ; % use std aggregation
param.do_mean = 0 ; % use mean aggregation
param.perpartL2 = 1 ; % normalize according to each part norm (from the training set)
param.partids = [1 2 3 4 5 6] ; % use "image_patch1" "left_hand" "right_hand" "upper_body" "full_body" "full_image" parts respectively
% param.partids = [1 2] ;

compute_my_cnn_features(param);
