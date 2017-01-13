function im = standardizeImage(im)
%rescale image to have height <= 400
im = im2single(im) ;
if size(im,1) > 400
    im = imresize(im, [400 NaN]) ;
end
