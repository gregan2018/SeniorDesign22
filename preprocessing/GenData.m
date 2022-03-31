unProcDir = '/Users/garrettregan/Downloads/val/Validation/';
cropDir = '/Users/garrettregan/Downloads/val/Validation_cropped/';
CEdir = '/Users/garrettregan/Downloads/val/Validation_CE/';
ext = '.png';

% List folders within cropped directory
folders = dir(unProcDir);
folderNames = {folders.name};
folders = folders(~startsWith(folderNames, '.'));
folderNames = {folders.name};
nFolders = size(folders, 1);

for dirIdx = 1:nFolders

    folderName = folderNames{dirIdx};
    disp(folderName);

    subFolders = dir([unProcDir, folderName]);
    subFolderNames = {subFolders.name};
    subFolderNames = subFolderNames(~startsWith(subFolderNames, '.'));
    nSubFolders = length(subFolderNames);

    for subIdx = 1:nSubFolders

        subFolderName = subFolderNames{subIdx};
        disp(subFolderName);

        files = dir([unProcDir, folderName, '/']);
        fileNames = {files.name};
        fileNames = fileNames(~startsWith(fileNames, '.'));
        nFiles = length(fileNames);

        for fIdx = 1:nFiles

            fileName = fileNames{fIdx};
            disp(fIdx);

            % read image and convert to grayscale
            img = imread([unProcDir, folderName, '/', fileName]);
            img = rgb2gray(img);

            % generate activations from cropping net
            approx = activations(net, img, 'FinalClassificationLayer');
            approx = approx(:,:,1) > 0.8;
            
            % morphological opening
            approx = imopen(approx, strel('disk', 5));

            % select largest obect
            objs = bwconncomp(approx, 4);
            sz = size(objs.PixelIdxList);
            sizes = zeros(sz);
    
            for k = 1:sz(2)
                s = size(objs.PixelIdxList{k});
                sizes(k) = s(1);
            end
    
            idx = find(sizes == max(sizes), 1, 'first');
            idxs = objs.PixelIdxList{idx};
            mask = false(size(img));
    
            for l = 1:length(idxs)
                mask(idxs(l)) = true;
            end
            
            % crop to the bounding box
            bbox = regionprops(mask, 'BoundingBox');
            bbox = bbox.BoundingBox;
            bbox = round(bbox);
    
            cols = bbox(1):bbox(1)+bbox(3);
            rows = bbox(2):bbox(2)+bbox(4);
            rows = min( max(rows, 1), 512);
            cols = min( max(cols, 1), 512);
            img = img(rows, cols);
            img = imresize(img, [512, 512]);

            % write cropped image
            imwrite(img, [cropDir, folderName, '/', extractBefore(fileName,'.'), '_crop.png']);

            % histogram match image using reference CDF
            img = HistMatch(img, cdf);

            % write the contrast enhanced image
            imwrite(img, [CEdir, folderName, '/', extractBefore(fileName,'.'), '_CE.png']);

        end
    end
end
