function rotateImageSet(path)
% scan a directory, copy and rotate each image from 72 to 288 (4 rotated versions) 
content = dir(path) ;
names = {content.name} ;
ok = regexpi(names, '.*\.(jpg|png|jpeg|gif|bmp|tiff)$', 'start') ;
names = names(~cellfun(@isempty,ok)) ; % filter the images with type shown above
for n = 1:length(names)
    ipath = fullfile(path,names{n}) ;
%     fprintf('%s\n',ipath);
    I = imread(ipath);
    for i = 72:72:288 
        I1 = imrotate(I,i,'nearest','crop');
        [~,s,e] = fileparts(ipath);
        baseFileName= strcat(s,sprintf('%s%d%s','Rotate',i,e));
        pathName=path;
        fullFileName = fullfile(pathName, baseFileName); % No need to worry about slashes
        imwrite(I1, fullFileName);
    end
end
