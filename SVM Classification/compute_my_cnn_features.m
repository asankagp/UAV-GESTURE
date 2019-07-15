
function compute_my_cnn_features(param)

% ----- PARAMETERS --------
% param.do_dyn = 1 ; % use dynamic features (differences)
% param.do_acc = 0 ; % use differences of dynamic features
% param.do_max = 1 ; % use max aggregation
% param.do_min = 1 ; % use min aggregation
% param.do_std = 0 ; % use std aggregation
% param.do_mean = 0 ; % use mean aggregation
% param.perpartL2 = 1 ; % normalize according to each part norm (from the training set)
% param.partids = [1 2 3 4 5 6] ; % use "image_patch1" "left_hand" "right_hand" "upper_body" "full_body" "full_image" parts respectively


if ~exist(param.savedir,'dir'); mkdir(param.savedir) ; end % create res folder if necessary

fprintf('\n------ Compute P-CNN features ------\n')

featdir_app_root=sprintf('%s/cnn_features_app/video_features/',param.cachepath);
featdir_flow_root=sprintf('%s/cnn_features_flow/video_features/',param.cachepath);

featdir_app_category = dir(featdir_app_root); 
featdir_flow_category = dir(featdir_flow_root);
feat_category = featdir_app_category(3:end);

num_category = length(feat_category);

for i = 1 : num_category
    
%     vidname = video_names{i};
    
    featdir_app = strcat(featdir_app_root,feat_category(i).name);
    featdir_flow = strcat(featdir_flow_root,feat_category(i).name);
    
    param.trainsplitpath = strcat('JHMDB/splits/',feat_category(i).name,'_train_split1.txt');
    param.testsplitpath = strcat('JHMDB/splits/',feat_category(i).name,'_test_split1.txt');
    
%     trainsplitpath = sprintf('JHMDB/splits/%s_train_split%d.txt',vidname,SplitT);
%     testsplitpath = sprintf('JHMDB/splits/%s_test_split%d.txt',vidname,SplitT);
%     
%     param.trainsplitpath = trainsplitpath;
%     param.testsplitpath = testsplitpath;
    
    disp('In appearance')
    if isfield(param,'perpartL2') && param.perpartL2
        fprintf('Compute per part norms --->  '); tic;
        norms=get_partnorms(param.trainsplitpath,featdir_app,param);
        fprintf('%d sec\n',round(toc));
    else
        norms=[];
    end
    
    [Xn_train,Xn_test] = get_Xn_train_test(featdir_app,param,norms);
    
    
    disp('In flow')
    if isfield(param,'perpartL2') && param.perpartL2
        fprintf('Compute per part norms --->  '); tic;
        norms=get_partnorms(param.trainsplitpath,featdir_flow,param);
        fprintf('%d sec\n',round(toc));
    else
        norms=[];
    end
    [Xn_trainOF,Xn_testOF] = get_Xn_train_test(featdir_flow,param,norms);
    
    Xn_train = cat(1,Xn_train,Xn_trainOF); clear Xn_trainOF ;
    Xn_test = cat(1,Xn_test,Xn_testOF); clear Xn_testOF ;
    
    if param.compute_kernel
        disp('Compute Kernel Test')
        Ktest   = Xn_test'*Xn_train;
        %savename=c('%s/Ktest.mat',param.savedir);
        savename = strcat(param.savedir,'/Ktest_',feat_category(i).name,'.mat');
        disp(['Save test kernel in: ',savename])
        assert(sum(isinf(Ktest(:)))==0 && sum(isnan(Ktest(:)))==0)
        save(savename,'Ktest','-v7.3')
        %savename=sprintf('%s/Xn_test.mat',param.savedir);
        savename = strcat(param.savedir,'/Xn_test_',feat_category(i).name,'.mat');
        disp(['Save test features in: ',savename])
        save(savename,'Xn_test','-v7.3')
        clear Ktest ;
        clear Xn_test ;
        
        disp('Compute Kernel Train')
        Ktrain  = Xn_train'*Xn_train;
        assert(sum(isinf(Ktrain(:)))==0 && sum(isnan(Ktrain(:)))==0)
        %savename=sprintf('%s/Ktrain.mat',param.savedir);
        savename = strcat(param.savedir,'/Ktrain_',feat_category(i).name,'.mat');
        disp(['Save train kernel in: ',savename])
        save(savename,'Ktrain','-v7.3')
        %savename=sprintf('%s/Xn_train.mat',param.savedir);
        savename = strcat(param.savedir,'/Xn_train_',feat_category(i).name,'.mat');
        disp(['Save train features in: ',savename])
        save(savename,'Xn_train','-v7.3')
        clear Ktrain ;
        clear Xn_train ;
        
        
    end
end
