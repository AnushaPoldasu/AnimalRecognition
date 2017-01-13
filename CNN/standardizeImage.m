function im = standardizeImage(im)
%rescale image to have a standard size of 256 pixels.
im = im2single(imread(im)) ;
s = 256/min(size(im,1),size(im,2)) ;
im = imresize(im, round(s*[size(im,1) size(im,2)])) ;
