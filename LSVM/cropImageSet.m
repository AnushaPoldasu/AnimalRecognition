function cropImageSet(path)
% crop the images in a given directory
content = dir(path) ;
names = {content.name} ;
ok = regexpi(names, '.*\.(jpg|png|jpeg|gif|bmp|tiff)$', 'start') ;
names = names(~cellfun(@isempty,ok)) ;
for n = 1:length(names)   
    ipath = fullfile(path,names{n}) ;
    fprintf('%s\n',ipath);
    I = imread(ipath);
    [~,s,~] = fileparts(ipath);
    I2 = imcrop(I);
    baseFileName = strcat(s,sprintf('%s%d.jpg', 'm',n)); % e.g. "1.png"
    Resultados=path;
    fullFileName = fullfile(Resultados, baseFileName); % No need to worry about slashes now!
    imwrite(I2, fullFileName);
end
