% Concatenate all features to create a new kernel
function accuracy = classification(param,SplitT)

% feats_root = 'p-cnn_features_split1/train_features';
feats_root = sprintf('p-cnn_features_split%d/train_features',SplitT);
savedir = param.savedir;
featsraw = dir(feats_root);
featsraw = featsraw(3:end);
temp = [];
L_train = [];
L_test = [];
for i = 1:length(featsraw)
    feats = sprintf('%s/%s',feats_root,featsraw(i).name);
    A = load(feats);
    A = A.Xn_train;
    m = size(A,2);
    L_train(end+1) = m;
    temp = [temp,A];  %temp=(PatchNum*4*4096*2)*(WholeDataset:TrainVideoClipNum);
end
filename = fullfile(savedir,'train_feats.mat');
save(filename,'temp');

Xn_train = temp;
clear temp

% feats_root = 'p-cnn_features_split1/test_features';
feats_root = sprintf('p-cnn_features_split%d/test_features',SplitT);
featsraw = dir(feats_root);
featsraw = featsraw(3:end);
temp = [];
for i = 1:length(featsraw)
    feats = sprintf('%s/%s',feats_root,featsraw(i).name); 
    A = load(feats);
    A = A.Xn_test;
    m = size(A,2);
    L_test(end+1) = m;
    temp = [temp,A];
end
Xn_test = temp;
filename = fullfile(savedir,'test_feats.mat');
save(filename,'temp');

% Ktrain = Xn_train'*Xn_train;
% Ktest = Xn_test'*Xn_train;

% save_svm = 'SVM';
% if ~exist(save_svm,'dir'); 
%     mkdir(save_svm); 
% end

n = size(Xn_train,2);
Labels_train = zeros(n,1);
counts = 0;
for j = 1: 13
    s = L_train(j); % Number of Train Clips of each action video
    for k = 1 : s
        Labels_train(k+counts,1) = j;
    end
    counts = counts + s;
end
% B = find(Labels_train ~=i);
% Labels_train(B) = -1;

model = svmtrain(Labels_train,Xn_train','-s 0 -t 0 -b 1');

filedir = sprintf('%s/splits%d_models',savedir,SplitT);
if ~exist(filedir,'dir'); 
    mkdir(filedir); 
end
filename = fullfile(filedir,'multi_model.mat');
save(filename,'model','Labels_train');

n = size(Xn_test,2);
Labels_test = zeros(n,1);
counts = 0;
for j = 1:13
    s = L_test(j);
    for k = 1 : s
        Labels_test(k+counts,1) = j;
    end
    counts = counts + s;
end
% B = find(Labels_test ~=i);
% Labels_test(B) = -1;

[predict_label,accuracy,dec_values] = svmpredict(Labels_test,Xn_test',model,'-b 1');
filename = fullfile(filedir,'multi_accuracy.mat');
save(filename,'predict_label','accuracy','dec_values','Labels_test');
