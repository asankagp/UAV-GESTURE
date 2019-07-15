clear all, close all

videopath = 'H:\Datasets\UAVACTION\Videos\';
savepath = 'H:\Datasets\UAVACTION\Images\';

classnames = dir(videopath);
classnames={classnames.name};
classnames=classnames(~ismember(classnames,{'.','..'}));
num_classname = length(classnames);

for cname = 1: num_classname
    classname = classnames{cname};
    
    vidnames = dir(strcat(videopath,classname));
    vidnames = {vidnames.name};
    vidnames = vidnames(~ismember(vidnames,{'.','..'}));
    num_vid = length(vidnames);
    
    for vname = 1:num_vid
        vidname = vidnames{vname};
        vidpath = strcat(videopath,classname,'\',vidname);
        
        vid=VideoReader(vidpath);
        numFrames = vid.NumberOfFrames;
        n=numFrames;
        for j = 1:n
            frame = read(vid,j);
            savedir = strcat(savepath,classname,'\',vidname(1:end-4),'\');
            
            if ~exist(savedir,'dir')
                mkdir(savedir);
            end
            
            framepath = sprintf('%d.png',j);
            framepath = strcat(savedir,framepath);
            imwrite(frame,framepath);
        end
    end
end