%% setup - add project folders to the path and load config
% Run once per MATLAB session (every script calls it).
projRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projRoot, 'src')));
addpath(fullfile(projRoot, 'scripts'));
cfg = config();
rng(cfg.seed);
