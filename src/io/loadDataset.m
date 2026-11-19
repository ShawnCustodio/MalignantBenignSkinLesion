function T = loadDataset(cfg)
%LOADDATASET Build a table of image/mask pairs with class labels.
%   T = loadDataset(cfg) scans cfg.dataDir/<classFolder>/ for ISIC_*.jpg and
%   their matching ISIC_*_Segmentation.png masks. Returns a table with
%   columns: id, label (categorical), imagePath, maskPath.

    ids = {}; labels = {}; imgs = {}; masks = {};
    for c = 1:numel(cfg.classFolders)
        folder = fullfile(cfg.dataDir, cfg.classFolders{c});
        if ~isfolder(folder)
            error('loadDataset:missingFolder', 'Class folder not found: %s', folder);
        end
        files = dir(fullfile(folder, 'ISIC_*.jpg'));
        for k = 1:numel(files)
            [~, id] = fileparts(files(k).name);
            maskPath = fullfile(folder, [id '_Segmentation.png']);
            if ~isfile(maskPath)
                warning('loadDataset:noMask', 'No mask for %s, skipping.', id);
                continue
            end
            ids{end+1,1}    = id;                              %#ok<AGROW>
            labels{end+1,1} = cfg.classNames{c};               %#ok<AGROW>
            imgs{end+1,1}   = fullfile(folder, files(k).name); %#ok<AGROW>
            masks{end+1,1}  = maskPath;                        %#ok<AGROW>
        end
    end

    T = table(string(ids), categorical(labels, cfg.classNames), string(imgs), string(masks), ...
        'VariableNames', {'id', 'label', 'imagePath', 'maskPath'});

    if numel(unique(T.id)) < height(T)
        warning('loadDataset:duplicates', 'Same ISIC id appears in more than one class folder.');
    end
end
