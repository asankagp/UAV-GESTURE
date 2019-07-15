function norms=get_partnorms(splitpath,featdirraw,param)
partids = param.partids;

%% Compute norms
[samplelist,numfil]=get_sample_list(splitpath,featdirraw);

norms=zeros(length(partids),numfil);
nframes=zeros(length(partids),numfil);
for ii=1:numfil
    pathname=samplelist{ii};
    tmp=load(pathname);
    norms_ii=norms(:,ii);
    nframes_ii=nframes(:,ii);
    for nd=1:length(partids)
        cnnf=tmp.features(partids(nd)).x;
        norms_ii(nd)=norms_ii(nd)+sum(sqrt(sum(cnnf.^2,2))); % Each clip
        nframes_ii(nd)=nframes_ii(nd)+size(cnnf,1);
    end
    norms(:,ii)=norms_ii;
    nframes(:,ii)=nframes_ii;
    
    %fprintf('NORM: %d out of %d\n',ii,numfil)
end
norms=sum(norms,2);   % Whole action video
nframes=sum(nframes,2);
norms=norms./nframes; % nframes = Num_clips(all clips) * Num_clip_frame(each clip contains frames)
