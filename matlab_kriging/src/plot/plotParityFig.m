function plotParityFig(yTrue, yPred, metrics, outputName, fontSize, expPath, expFmt)
arguments
    yTrue      (:,1) double
    yPred      (:,1) double
    metrics    (1,1) struct
    outputName (1,1) string = "Output"
    fontSize   (1,1) double = 11
    expPath    (1,1) string = ""
    expFmt     (1,1) string = "png"
end
fig = figure("Name", "Parity — " + outputName, "NumberTitle", "off");
tl  = tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");

lo = min([yTrue; yPred]);
hi = max([yTrue; yPred]);
if abs(hi - lo) < eps, lo = lo - 1; hi = hi + 1; end

%% 1) Parity
ax1 = nexttile(tl, 1);
scatter(ax1, yTrue, yPred, 42, "filled", ...
    "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.65);
hold(ax1, "on");
plot(ax1, [lo hi], [lo hi], "r--", "LineWidth", 2);
xlabel(ax1, "True",      "FontSize", fontSize);
ylabel(ax1, "Predicted", "FontSize", fontSize);
title(ax1, outputName + " parity", "FontSize", fontSize, "Interpreter", "none");
axis(ax1, "equal"); xlim(ax1, [lo hi]); ylim(ax1, [lo hi]);
grid(ax1, "on");

%% 2) Absolute error vs true
ax2 = nexttile(tl, 2);
scatter(ax2, yTrue, metrics.absResiduals, 42, "filled", ...
    "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.65);
hold(ax2, "on");
yline(ax2, metrics.mae,   "b--", sprintf("MAE = %.3g",   metrics.mae),   ...
    "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
yline(ax2, metrics.maxAE, "r--", sprintf("MaxAE = %.3g", metrics.maxAE), ...
    "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
xlabel(ax2, "True value",      "FontSize", fontSize);
ylabel(ax2, "Absolute error",  "FontSize", fontSize);
title(ax2,  "Absolute error",  "FontSize", fontSize);
grid(ax2, "on");

%% 3) Percentage error vs true
ax3 = nexttile(tl, 3);
scatter(ax3, yTrue, metrics.percResiduals, 42, "filled", ...
    "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.65);
hold(ax3, "on");
yline(ax3, 0, "k-", "LineWidth", 1.0, "HandleVisibility", "off");
yline(ax3,  metrics.mape, "g--", sprintf("MAPE = %.2f%%",  metrics.mape), ...
    "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
yline(ax3, -metrics.mape, "g--", "LineWidth", 1.6, "HandleVisibility", "off");
xlabel(ax3, "True value",              "FontSize", fontSize);
ylabel(ax3, "Percentage error [%]",    "FontSize", fontSize);
title(ax3,  "Signed percentage error", "FontSize", fontSize);
grid(ax3, "on");

%% 4) Percentage error histogram
ax4 = nexttile(tl, 4);
histogram(ax4, metrics.percResiduals, 25, "EdgeColor", "k", "FaceAlpha", 0.75);
hold(ax4, "on");
xline(ax4,  metrics.mape,   "g--", sprintf("MAPE = %.2f%%",   metrics.mape),   ...
    "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
xline(ax4, -metrics.mape,   "g--", "LineWidth", 1.6, "HandleVisibility", "off");
xline(ax4,  metrics.maxAPE, "r--", sprintf("MaxAPE = %.2f%%", metrics.maxAPE), ...
    "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
xline(ax4, -metrics.maxAPE, "r--", "LineWidth", 1.6, "HandleVisibility", "off");
xlabel(ax4, "Percentage error [%]", "FontSize", fontSize);
ylabel(ax4, "Count",                "FontSize", fontSize);
title(ax4,  "Percentage error histogram", "FontSize", fontSize);
grid(ax4, "on");

title(tl, sprintf("%s | RMSE=%.4g | R²=%.4f | MAPE=%.2f%%", ...
    outputName, metrics.rmse, metrics.r2, metrics.mape), ...
    "FontSize", fontSize + 2, "Interpreter", "none");

exportFigure(fig, "Parity_" + sanitizeFilename(outputName), expPath, expFmt);
end
