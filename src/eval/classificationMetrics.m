function m = classificationMetrics(yTrue, yPred, positiveClass)
%CLASSIFICATIONMETRICS Accuracy, sensitivity, specificity, precision, F1.
    yTrue = categorical(yTrue); yPred = categorical(yPred);
    pos = (yTrue == positiveClass); predPos = (yPred == positiveClass);
    TP = sum(pos & predPos);   FN = sum(pos & ~predPos);
    TN = sum(~pos & ~predPos); FP = sum(~pos & predPos);

    m.accuracy    = (TP + TN) / numel(yTrue);
    m.sensitivity = TP / max(TP + FN, 1);
    m.specificity = TN / max(TN + FP, 1);
    m.precision   = TP / max(TP + FP, 1);
    m.f1          = 2 * m.precision * m.sensitivity / max(m.precision + m.sensitivity, eps);
    m.confusion   = [TP FN; FP TN];
end
