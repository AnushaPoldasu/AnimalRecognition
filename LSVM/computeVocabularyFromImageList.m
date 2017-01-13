function vocabulary = computeVocabularyFromImageList(names)
% compute a visual word vocabulary from a list of image names (paths),
% which contains words and kdtree indexing the visual word for fast quantization.

numWords = 1000;
numFeatures = numWords * 100;

% extracts visual descriptors 
descriptors = cell(1,numel(names));
for i = 1:numel(names)
  fullPath = names{i} ;
  fprintf('Extracting features from %s\n', fullPath);
  im = imread(fullPath);
  im = standardizeImage(im);
  %compute features
  [~, descriptor] = vl_phow(im, 'step', 4, 'floatdescriptors', true); % extract Dense SIFT features from 4 pixels of grid 
  descriptors{i} = vl_colsubset(descriptor, round(numFeatures / numel(names)), 'uniform'); % randomly choose numFeature/numImages features
end

% cluster the descriptors into visual words using kmeans algorithm
% compute a KDTREE to index them to speeds-up quantization significantly.
fprintf('Computing visual words and kdtree\n');
descriptors = single([descriptors{:}]);
vocabulary.words = vl_kmeans(descriptors, numWords, 'verbose', 'algorithm', 'elkan');
vocabulary.kdtree = vl_kdtreebuild(vocabulary.words);

