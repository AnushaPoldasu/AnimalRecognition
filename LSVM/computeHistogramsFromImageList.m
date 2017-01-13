function histograms = computeHistogramsFromImageList(vocabulary, names)
% compute historams for multiple images
    histograms = cell(1,numel(names));
    for i = 1:length(names)
        fullPath = names{i};
        fprintf('Extracting histogram from %s\n' , fullPath);
        histograms{i} = computeHistogramFromImage(vocabulary, fullPath);
    end
    histograms = [histograms{:}];

    % compute the histogram of visual words for image given the visual word vocaublary VOCABULARY.
    function histogram = computeHistogramFromImage(vocabulary,fullPath)
        im = imread(fullPath);
        height= size(im,1);
        width = size(im,2);
        im = standardizeImage(im);
        % compute features:  
        [keypoints, descriptors] = vl_phow(im, 'step', 4, 'floatdescriptors', true) ;% extract Dense SIFT features from 4 pixels
        % quantize visual descriptors into visual words using approximated nearest-neighbors
        [words,~] = vl_kdtreequery(vocabulary.kdtree, vocabulary.words, descriptors, 'MaxComparisons', 20) ; % at most 20comparisons per query point
        words = double(words) ;
        %compute histogram
        histogram = computeHistogram(width, height, keypoints, words) ;
    end

    % computes a 2x2 spatial histogram of the N visual words then do geometric tiling
    function histogram = computeHistogram(imagewidth, imageheight, keypoints, words)
        numWords = 1000 ;
        numSpatialX = 2 ;
        numSpatialY = 2 ;

        % map corrdinates to the bins 
        binsx = vl_binsearch(linspace(1,imagewidth, numSpatialX+1), keypoints(1,:)); % x coordinate
        binsy = vl_binsearch(linspace(1,imageheight,numSpatialY+1), keypoints(2,:)); % y coordinate
        % converts the subscripts (y, x, words) for three-dimensional array to a single linear index.
        bins = sub2ind([numSpatialY, numSpatialX, numWords], binsy,binsx,words); % get all indexes
        histile = zeros(numSpatialY * numSpatialX * numWords, 1) ; % empty histogram
        histile = vl_binsum(histile, ones(size(bins)), bins) ; % add the weight to make histogram
        histogram = single(histile/sum(histile)); %single precision: 32 bit instead of 64 bit for double 
    end
end
