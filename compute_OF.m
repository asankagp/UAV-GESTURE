function compute_OF(video_names,param)

% create cache folder
dname=sprintf('%s/%s',param.cachepath,'OF');
if ~exist(dname,'dir'); mkdir(dname) ; end

fprintf('\n------ Compute OF ------\n')

nb_vid=length(video_names);
impath=param.impath;
cachepath=param.cachepath;
imext=param.imext;

for vi = 1:nb_vid
    vidname=video_names{vi} ;
    
    subvideo_names = dir(sprintf('%s/%s',impath,vidname));
    subvideo_names = {subvideo_names.name};
    subvideo_names = subvideo_names(~ismember(subvideo_names,{'.','..'}));
    subvideo_names = natsortfiles(subvideo_names);
    subnb_vid = length(subvideo_names);
    
    for subvi = 1:subnb_vid
        subvidname = subvideo_names{subvi};
        
        images=dir(sprintf('%s/%s/%s/*%s',impath,vidname,subvidname,imext));
        images = {images.name};
        images = natsortfiles(images);
        
        savedir = sprintf('%s/OF/%s/%s',cachepath,vidname,subvidname) ;
        if ~exist(savedir,'dir'); mkdir(savedir) ; end
        
        fprintf('compute OF: %d out of %d videos in category %d\n',subvi,subnb_vid,vi)
        parfor i=1:length(images)-1
            
            [~,imname,~]=fileparts(images{i});
            imsave=sprintf('%s/%s.jpg',savedir,imname);
            if exist(imsave,'file') ; continue ; end
            
            im1 = double(imread(sprintf('%s/%s/%s/%s',impath,vidname,subvidname,images{i})));
            im2 = double(imread(sprintf('%s/%s/%s/%s',impath,vidname,subvidname,images{i+1})));
            
            max_flow = 8; % maximum absolute value of flow
            scalef = 128/max_flow;
            
            flow = mex_OF(im1,im2); % FROM THOMAS BROX 2004
            
            x = flow(:,:,1); y = flow(:,:,2);
            if range(x(:))<0.5 && range(y(:))<0.5 % some frames are duplicate; ignore flow for those as it will be confusing for the CNN
                continue;
            end
            
            mag_flow = sqrt(sum(flow.^2,3));
            
            %max(flow,[],'all')
            %min(flow,[],'all')
            
            flow = flow*scalef;  % scale flow
            flow = flow+128;    % center it around 128
            flow(flow<0) = 0;
            flow(flow>255) = 255; % crop the values below 0 and above 255
            
            mag_flow = mag_flow*scalef; % same for magnitude
            mag_flow = mag_flow+128;
            mag_flow(mag_flow<0) = 0;
            mag_flow(mag_flow>255) = 255;
            
            im = uint8(cat(3,flow,mag_flow)); % concatenate flow_x, flow_y and magnitude
            imwrite(im,imsave);
        end
    end
end
