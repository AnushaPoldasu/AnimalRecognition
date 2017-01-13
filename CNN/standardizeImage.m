function im = standardizeImage(im)
%rescale image to 256* 256 pixels.
im = im2single(imread(im)) ;
s = 256/max(size(im,1),size(im,2)) ;
im = imresize(im, round(s*[size(im,1) size(im,2)])) ;
