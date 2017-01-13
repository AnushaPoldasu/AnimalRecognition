
function disc = computediscFromImageList(encoder, names)

    if numel(names)>1
        disc = cell(1,numel(names));
    end
    for in = 1:length(names)
       fullPath = names{in} ;
       disc{in} = encodeImage(encoder, fullPath);
    end
    disc = [disc{:}] ;

    function disc = encodeImage(encoder, im)
        if ~iscell(im), im = {im} ; end
        disc = cell(1,numel(im)) ;
        if numel(im) > 1
          for i = 1:numel(im)
            disc{i} = encodeOne(encoder, im{i}) ;
          end
        elseif numel(im) == 1
          disc{1} = encodeOne(encoder, im{1}) ;
        end
        disc = cat(2, disc{:}) ;
    end


    % --------------------------------------------------------------------
    function disc = encodeOne(encoder, im)
    % --------------------------------------------------------------------
        fprintf('encoding image from %s\n', im) ;
        im = standardizeImage(im) ;
        im_ = bsxfun(@minus, 255*im, encoder.averageColor) ;
        res = vl_simplenn(encoder.net, im_) ;
        disc = mean(reshape(res(end).x, [], size(res(end).x,3)), 1)' ;
    end
end