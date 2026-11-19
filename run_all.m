%% run_all - run the complete pipeline from scratch (~25 min)
% Each step saves its outputs in results/, so steps can also be run alone
% once the earlier ones have finished.
setup;
t0 = tic;
steps = {'s00_check_data', 's01_preprocess', 's02_features', ...
         's03_segment', 's04_build_graphs', 's05_classify'};
for k = 1:numel(steps)
    fprintf('\n################ %s ################\n', steps{k});
    run(steps{k});
    close all;
end
fprintf('\nAll steps finished in %.1f min. See results/ and results/figures/.\n', toc(t0) / 60);
