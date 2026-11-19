%% Step 0 - Sanity-check the dataset
% Verifies every image has a mask, sizes match, masks are non-empty, and
% shows a montage of each class with the mask outline overlaid.

setup;
T = loadDataset(cfg);

fprintf('Data folder: %s\n', cfg.dataDir);
disp(groupcounts(T, 'label'));

n = height(T);
imgH = zeros(n,1); imgW = zeros(n,1); lesionFrac = zeros(n,1); sizeMismatch = false(n,1);
for i = 1:n
    info = imfinfo(T.imagePath(i)); minfo = imfinfo(T.maskPath(i));
    imgH(i) = info.Height; imgW(i) = info.Width;
    sizeMismatch(i) = info.Height ~= minfo.Height || info.Width ~= minfo.Width;
    [~, M] = readSample(T, i);
    lesionFrac(i) = mean(M(:));
end
T.height = imgH; T.width = imgW; T.lesionFraction = lesionFrac;

fprintf('Image sizes: %d unique (H %d-%d, W %d-%d)\n', ...
    size(unique([imgH imgW], 'rows'), 1), min(imgH), max(imgH), min(imgW), max(imgW));
fprintf('Image/mask size mismatches: %d\n', sum(sizeMismatch));
fprintf('Empty masks: %d\n', sum(lesionFrac == 0));
fprintf('Lesion area fraction: median %.2f (min %.2f, max %.2f)\n', ...
    median(lesionFrac), min(lesionFrac), max(lesionFrac));

if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
writetable(T, fullfile(cfg.resultsDir, 'dataset_index.csv'));

% Visual check: first 8 images per class with mask outline
if usejava('desktop')
    for c = categories(T.label)'
        idx = find(T.label == c{1}, 8);
        tiles = cell(numel(idx), 1);
        for k = 1:numel(idx)
            [I, M] = readSample(T, idx(k));
            tiles{k} = imresize(imoverlay(I, bwperim(imdilate(M, strel('disk', 3))), [0 1 0]), [256 256]);
        end
        figure('Name', c{1}); montage(tiles, 'Size', [2 4]); title(c{1});
    end
end
