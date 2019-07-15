function extract_cnn_patches(video_names,param)

% create cache folders
cdirs={'patches_app','patches_flow','patches_app/left_hand','patches_flow/left_hand', ...
    'patches_app/right_hand','patches_flow/right_hand','patches_app/upper_body','patches_flow/upper_body', ...
    'patches_app/full_body','patches_flow/full_body','patches_app/full_image','patches_flow/full_image'};
for d=1:length(cdirs)
    dname=sprintf('%s/%s',param.cachepath,cdirs{d});
    if ~exist(dname,'dir'); mkdir(dname) ; end
end

fprintf('\n------ Compute CNN patches ------\n')

for vi = 1:length(video_names)
    fprintf('extract patches .. : %d out of %d videos\n',vi,length(video_names))
    
    % get image list in the current video
    vidname=video_names{vi} ;
    
    % Subsequences Directory
    for d=3:length(cdirs)
        dname=sprintf('%s/%s',param.cachepath,cdirs{d});
        subdname=sprintf('%s/%s',dname,vidname);
        if ~exist(subdname,'dir')
            mkdir(subdname);
        end
    end
    
    subvideo_names = dir(sprintf('%s/%s',param.impath,vidname));
    subvideo_names = {subvideo_names.name};
    subvideo_names = subvideo_names(~ismember(subvideo_names,{'.','..'}));
    subvideo_names = natsortfiles(subvideo_names);
    subnb_vid = length(subvideo_names);
    
    for subvi = 1:subnb_vid
        subvidname = subvideo_names{subvi};
        
        images=dir(sprintf('%s/%s/%s/*%s',param.impath,vidname,subvidname,param.imext));
        images = {images.name};
        images = natsortfiles(images);
        
        % get video joint positions
        positions=load(sprintf('%s/%s/%s/%s',param.jointpath,vidname,subvidname,subvidname)) ;
        positions=positions.pos_img ;
        
        suf={'app','flow'} ;
        imdirs = {param.impath,sprintf('%s/OF',param.cachepath)};
        
        for i=1:2 % appearance and flow
            imdirpath = imdirs{i};
            
            net=param.(sprintf('net_%s',suf{i}));
            
            for idim=1:min(length(images),length(positions))
                if exist(sprintf('%s/full_image/%s/%s_im%05d.jpg',param.cachepath,vidname,subvidname,idim),'file')
                    continue;
                end
                
                % get image
                if i==1 % appearance
                    impath = sprintf('%s/%s/%s/%s',imdirpath,vidname,subvidname,images{idim}) ;
                else % flow
                    [~,iname,~]=fileparts(images{idim});
                    impath = sprintf('%s/%s/%s/%s.jpg',imdirpath,vidname,subvidname,iname) ; % flow has been previously saved in JPG
                    if ~exist(impath,'file'); continue ; end ; % flow was not computed (see compute_OF.m for info)
                end
                im = imread(impath);
                
                % get part boxes
                % part CNN (fill missing part before resizing)
                lside=param.lside;
                
                % left hand
                if positions(1,param.lhandposition,idim)==0 %fill wiith min values to avoid zeros
                    if idim~=1
                        positions(:,param.lhandposition,idim) = positions(:,param.lhandposition,idim-1);
                    else
                        positions(1,param.lhandposition,idim) = min(nonzeros(positions(1,:,idim)));
                        positions(2,param.lhandposition,idim) = min(nonzeros(positions(2,:,idim)));
                    end
                end
                
                lhand = get_box_and_fill(positions(:,param.lhandposition,idim)-lside,positions(:,param.lhandposition,idim)+lside,im);
                lhand = imresize(lhand, net.normalization.imageSize(1:2)) ;
                
                % right right
                if positions(1,param.rhandposition,idim)==0
                    if idim~=1
                        positions(:,param.rhandposition,idim) = positions(:,param.rhandposition,idim-1);
                    else
                        positions(1,param.rhandposition,idim) = min(nonzeros(positions(1,:,idim)));
                        positions(2,param.rhandposition,idim) = min(nonzeros(positions(2,:,idim)));
                    end
                end
                
                rhand = get_box_and_fill(positions(:,param.rhandposition,idim)-lside,positions(:,param.rhandposition,idim)+lside,im);
                rhand = imresize(rhand, net.normalization.imageSize(1:2)) ;
                
                % upper body
                for jo = 1:18
                    if positions(1,jo,idim)==0
                        positions(1,jo,idim) = min(nonzeros(positions(1,:,idim)));
                        positions(2,jo,idim) = min(nonzeros(positions(2,:,idim)));
                    end
                end
                
                lside=3/4*param.lside;
                upbody = get_box_and_fill(min(positions(:,param.upbodypositions,idim),[],2)-lside,max(positions(:,param.upbodypositions,idim),[],2)+lside,im);
                upbody = imresize(upbody, net.normalization.imageSize(1:2)) ;
                
                % full body
                if positions(1,11,idim)==0
                    positions(1,11,idim) = max(positions(1,:,idim));
                    positions(2,11,idim) = max(positions(2,:,idim));
                end
                
                
                fullbody = get_box_and_fill(min(positions(:,:,idim),[],2)-lside,max(positions(:,:,idim),[],2)+lside,im);
                fullbody = imresize(fullbody, net.normalization.imageSize(1:2)) ;
                
                % full image CNNf (just resize frame)
                fullim = imresize(im, net.normalization.imageSize(1:2)) ;
                
                imwrite(lhand,sprintf('%s/patches_%s/left_hand/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
                imwrite(rhand,sprintf('%s/patches_%s/right_hand/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
                imwrite(upbody,sprintf('%s/patches_%s/upper_body/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
                imwrite(fullbody,sprintf('%s/patches_%s/full_body/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
                imwrite(fullim,sprintf('%s/patches_%s/full_image/%s/%s_im%05d.jpg',param.cachepath,suf{i},vidname,subvidname,idim));
            end
        end
    end
end
