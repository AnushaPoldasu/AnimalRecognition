function  randomSplitImgFile(spath, dpath1,dpath2,num)
% given spath and negapath for all possible train data
% randomly choose number train data from each path and combined into a file
% step 1 clear both dpath1 and dpath2 folders
delete(fullfile(dpath1,'*.*'))
delete(fullfile(dpath2,'*.*'));
% step 2 copy all files from spath to dpath1 (assume spath has 500 images)
copyfile(spath,dpath1);
% step 3 randomly move num files to dpath2 (num depends on the ratio you want)

d1 = dir(dpath1);
n1 = randperm(length(d1)-3,num)+3; % consider the non image files
for i = 1:num
    Filename = d1(n1(i)).name;
    fullpath = fullfile(dpath1,Filename) ;
    movefile(fullpath,dpath2)
end


