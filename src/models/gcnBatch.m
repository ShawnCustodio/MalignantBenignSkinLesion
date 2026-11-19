function [A, X, Pool] = gcnBatch(graphs, model)
%GCNBATCH Pack several graphs into one disconnected graph (block-diagonal A).
    nodes = arrayfun(@(g) size(g.X, 1), graphs);
    if ~isfield(model, 'useEdges') || model.useEdges
        A = dlarray(single(full(blkdiag(graphs.Ahat))));
    else
        A = dlarray(eye(sum(nodes), 'single'));     % ablation: no neighbors
    end
    X = dlarray(single((vertcat(graphs.X) - model.mu) ./ model.sd));
    B = numel(graphs); N = sum(nodes);
    graphOf = repelem((1:B)', nodes);
    Pool = single(full(sparse(graphOf, (1:N)', 1 ./ nodes(graphOf), B, N)));
end
