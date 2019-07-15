function Xn=collect_samples(splitpath,featdirraw,param,norms) % norms is the normalized features (app or flow) in each channel (P1, P2, P3, ...)
do_dyn = param.do_dyn ;
do_acc = param.do_acc ;
do_min = param.do_min ;
do_std = param.do_std ;
do_mean =param.do_mean;
partids =param.partids;
partL2=isfield(param,'perpartL2') && param.perpartL2 ;

if isfield(param,'do_max')
    do_max = param.do_max ;
else
    do_max = 1 ; % do maximum aggregation by default
end

assert(~(~do_min && ~do_max && ~do_mean && ~do_std));

%% Load features

% get sample list
[samplelist,numfil]=get_sample_list(splitpath,featdirraw);

% pre-allocate memory
tmp=load(samplelist{1}); % One clip
Xn=zeros((do_dyn+do_acc+1)*(do_min+do_std+do_mean+do_max)*length(partids)*length(tmp.features(1).x(1,:)),numfil);
if partL2
    invrepnorms=1./repmat(norms',length(tmp.features(1).x(1,:)),1);invrepnorms=invrepnorms(:)';
else
    invrepnorms=[];
end

for ii=1:numfil
    pathname=samplelist{ii}; % Each video clip
    tmp=load(pathname);
    cnnf=[ tmp.features(partids).x ];
    
    if partL2
        cnnf=bsxfun(@times,cnnf,invrepnorms); % cnnf=cnnf*invrepnorms; Normalization
    end
    
    cnnf_diff=[];cnnf_acc=[];maxV = [];minV = [];stdV = [];meanV=[];
    if do_dyn
        if size(cnnf,1)>3;
            cnnf_diff = cnnf(4:end,:) - cnnf(1:end-3,:);
        elseif size(cnnf,1)>1;
            cnnf_diff = cnnf(2:end,:) - cnnf(1:end-1,:);
        else
            cnnf_diff=zeros(size(cnnf));
        end
        
        if do_acc
            if size(cnnf_diff,1)>1
                cnnf_acc = cnnf_diff(2:end,:) - cnnf_diff(1:end-1,:);
            else
                cnnf_acc=zeros(size(cnnf));
            end
        end
    end
    
    if do_max
        maxV = [max(cnnf,[],1)' ; max(cnnf_diff,[],1)' ; max(cnnf_acc,[],1)']; % max(cnnf,[],1) = PatchNum*4096; For each feature position, max(f1,f2,...fn)
    end
    if do_min
        minV = [min(cnnf,[],1)' ; min(cnnf_diff,[],1)' ; min(cnnf_acc,[],1)'];
    end
    if do_std
        stdV = [std(cnnf,0,1)' ; std(cnnf_diff,0,1)'; std(cnnf_acc,0,1)'];
    end
    if do_mean
        meanV = [mean(cnnf,1)' ; mean(cnnf_diff,1)' ; mean(cnnf_acc,1)'];
    end
    
    Xn(:,ii)=[maxV ; minV ; stdV ; meanV];  %Xn(:,ii)=PatchNum*4*4096;
    
    % fprintf('%d out of %d\n',ii,numfil)
end
