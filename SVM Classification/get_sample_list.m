function [samplelist,numfil]=get_sample_list(splitpath,featdirraw)
% open image list
split = fopen(splitpath) ;

% pre-allocate memory
%setenv('filepath',splitpath);
%[~,numfil]=system('cat $filepath | wc -l');numfil=str2num(numfil);
%samplelist=cell(numfil,1) ;
samplelist=cell(1000,1) ;

[sample,~] = strtok(fgetl(split));
ii=0; % number of loaded samples
while ischar(sample)
    ii=ii+1;
    
    if ii > length(samplelist) % allocate more
        samplelist=cat(1,samplelist,cell(1000,1));
    end
    
    r = strrep(sample,'.avi','');
    samplelist{ii}=[featdirraw '/' r '.mat'];
    %fprintf('Collect Sample: %d out of %d : %s\n',ii,numfil,sample)
    [sample,~] = strtok(fgetl(split));
end
fclose(split);

samplelist=samplelist(1:ii);
numfil=ii;
%assert(numfil == ii);
