
function psis = computepsisFromImageList(encoder, names)

    if numel(names)>1
        psis = cell(1,numel(names));
    end
    for i = 1:length(names)
       fullPath = names{i} ;
       psis{i} = encodeImage(encoder, fullPath);
    end
    psis = [psis{:}] ;

    function psi = encodeImage(encoder, im)
        if ~iscell(im), im = {im} ; end
        psi = cell(1,numel(im)) ;
        if numel(im) > 1
          for i = 1:numel(im)
            psi{i} = encodeOne(encoder, im{i}) ;
          end
        elseif numel(im) == 1
          psi{1} = encodeOne(encoder, im{1}) ;
        end
        psi = cat(2, psi{:}) ;
    end


    % --------------------------------------------------------------------
    function psi = encodeOne(encoder, im)
    % --------------------------------------------------------------------
        fprintf('encoding image from %s\n', im) ;
        im = standardizeImage(im) ;
        im_ = bsxfun(@minus, 255*im, encoder.averageColor) ;
        res = vl_simplenn(encoder.net, im_) ;
        psi = mean(reshape(res(end).x, [], size(res(end).x,3)), 1)' ;
    end
end