function [Xn_train,Xn_test] = get_Xn_train_test(featdirraw,param,norms) %norms(1)="image_patch1";norms(1)="image_patch2";norms(3)="full_image"
fprintf('Collect train samples --->  '); tic;
Xn_train=collect_samples(param.trainsplitpath,featdirraw,param,norms) ;
fprintf('%d sec\n',round(toc));

fprintf('Collect test samples --->  '); tic;
Xn_test=collect_samples(param.testsplitpath,featdirraw,param,norms);
fprintf('%d sec\n',round(toc));
