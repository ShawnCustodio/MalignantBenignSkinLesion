function model = gcnTrain(graphs, isMalignant, opts)
%GCNTRAIN Train the graph classifier with a custom loop (Deep Learning Toolbox).
%   graphs      : struct array from buildLesionGraph (training fold only)
%   isMalignant : logical vector, one per graph
%   opts        : cfg.gcn
%   Node features are standardized with mean/std of the TRAINING nodes only.

    %   opts.useEdges = false gives the ABLATION model: identical network but
    %   every region only sees itself (A = I), i.e. no spatial neighbor info.
    if ~isfield(opts, 'useEdges'), opts.useEdges = true; end
    model.useEdges = opts.useEdges;

    Xall = vertcat(graphs.X);
    model.mu = mean(Xall, 1);
    model.sd = std(Xall, 0, 1) + 1e-6;

    d = size(Xall, 2); h = opts.hidden;
    glorot = @(a, b) dlarray(single((rand(a, b) * 2 - 1) * sqrt(6 / (a + b))));
    params.W1  = glorot(d, h); params.b1  = dlarray(zeros(1, h, 'single'));
    params.W2  = glorot(h, h); params.b2  = dlarray(zeros(1, h, 'single'));
    params.Wfc = glorot(h, 2); params.bfc = dlarray(zeros(1, 2, 'single'));

    Y = single([~isMalignant(:) isMalignant(:)]);
    n = numel(graphs);
    avg = []; avgSq = []; iter = 0;
    model.lossCurve = zeros(opts.epochs, 1);
    for epoch = 1:opts.epochs
        perm = randperm(n);
        epochLoss = 0;
        for b0 = 1:opts.batch:n
            idx = perm(b0:min(b0 + opts.batch - 1, n));
            [A, X, Pool] = gcnBatch(graphs(idx), model);
            iter = iter + 1;
            [loss, grads] = dlfeval(@lossFcn, params, A, X, Pool, Y(idx, :), opts.dropout);
            for f = ["W1" "W2" "Wfc"]               % weight decay on weights only
                grads.(f) = grads.(f) + opts.l2 * params.(f);
            end
            [params, avg, avgSq] = adamupdate(params, grads, avg, avgSq, iter, opts.lr);
            epochLoss = epochLoss + double(extractdata(loss)) * numel(idx);
        end
        model.lossCurve(epoch) = epochLoss / n;
    end
    model.params = params;
end

function [loss, grads] = lossFcn(params, A, X, Pool, Y, dropoutRate)
    logits = gcnForward(params, A, X, Pool, dropoutRate, true);
    m = max(logits, [], 2);
    logProb = logits - (m + log(sum(exp(logits - m), 2)));
    loss = -mean(sum(Y .* logProb, 2));
    grads = dlgradient(loss, params);
end
