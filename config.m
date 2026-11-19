function cfg = config()
%CONFIG Central project settings. Every script/function reads paths from here.
%
%   The dataset is NOT stored in the repo. Point cfg.dataDir at the folder
%   that contains the Benign/ and malignant/ subfolders, either by editing
%   the default below or by setting the environment variable SKIN_DATA_DIR
%   (lets each teammate keep the data wherever they like).

    root = fileparts(mfilename('fullpath'));
    cfg.root = root;

    envDir = getenv('SKIN_DATA_DIR');
    if ~isempty(envDir)
        cfg.dataDir = envDir;
    elseif isfolder(fullfile(root, 'data', 'SkinCancer'))
        cfg.dataDir = fullfile(root, 'data', 'SkinCancer');
    else
        cfg.dataDir = 'C:\Users\shawn\Downloads\SkinCancer\SkinCancer';
    end

    % Subfolder name -> class label (folder names are case-sensitive on Linux/macOS)
    cfg.classFolders = {'Benign', 'malignant'};
    cfg.classNames   = {'benign', 'malignant'};
    cfg.positiveClass = 'malignant';   % for sensitivity/specificity

    cfg.resultsDir = fullfile(root, 'results');

    % Preprocessing
    cfg.pre.shortSide   = 512;         % resize so the shorter side = 512 px (keeps aspect ratio,
                                       % so shape/asymmetry features are not distorted)
    cfg.pre.hairLineLen = 15;          % length (px) of line elements for hair detection
    cfg.pre.hairThresh  = 0.045;       % black-top-hat threshold (0-1 intensity)
    cfg.pre.illumSigma  = 60;          % Gaussian sigma (px) of the illumination field
    cfg.pre.grayWorldP  = 6;           % Minkowski norm for Shades-of-Gray color constancy
    cfg.pre.clipLimit   = 0.01;        % CLAHE clip limit (texture image only)

    % Graph construction (superpixels inside the lesion = graph nodes)
    cfg.graph.nodesPerLesion = 150;    % target number of superpixels inside the lesion
    cfg.graph.compactness    = 10;     % SLIC compactness (higher = more regular regions)

    % GCN
    % Only ~52 training graphs per fold, so the network is kept small and
    % heavily regularized. Fixed a priori (NOT tuned on test folds - that
    % would need nested CV): a first trial with 32 units / 150 epochs drove
    % training loss to ~0.006, i.e. memorization.
    cfg.gcn.hidden   = 16;             % hidden units per graph-conv layer
    cfg.gcn.epochs   = 100;
    cfg.gcn.lr       = 0.005;          % Adam learning rate
    cfg.gcn.batch    = 8;              % graphs per mini-batch
    cfg.gcn.dropout  = 0.5;
    cfg.gcn.l2       = 1e-3;           % weight decay

    % Random Forest
    cfg.rf.numTrees = 300;
    cfg.rf.minLeaf  = 1;

    % Evaluation
    cfg.kFolds    = 5;                 % stratified k-fold CV
    cfg.cvRepeats = 5;                 % repeat CV with different splits (small data -> noisy)
    cfg.seed      = 42;                % reproducibility
end
