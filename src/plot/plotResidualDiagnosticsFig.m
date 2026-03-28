function plotResidualDiagnosticsFig(yTrue, yPred, metrics, outputName, fontSize, expPath, expFmt)
arguments
    yTrue      (:,1) double
    yPred      (:,1) double
    metrics    (1,1) struct
    outputName (1,1) string = "Output"
    fontSize   (1,1) double = 11
    expPath    (1,1) string = ""
    expFmt     (1,1) string = "png"
end
res        = metrics.residuals;
absRes     = metrics.absResiduals;
percRes    = metrics.percResiduals;
absPercRes = abs(percRes);

p68a = prctile(absRes,     68);   p99a = prctile(absRes,     99);
p68p = prctile(absPercRes, 68);   p99p = prctile(absPercRes, 99);

fig = figure("Name", "Residuals — " + outputName, "NumberTitle", "off");
tl  = tiledlayout(fig, 2, 3, "TileSpacing", "compact", "Padding", "compact");

%% 1) Residuals vs predicted
ax1 = nexttile(tl, 1);
scatter(ax1, yPred, res, 38, "filled", ...
    "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6);
hold(ax1, "on");
yline(ax1, 0,    "r--", "LineWidth", 1.5);
yline(ax1,  p68a, "g--", sprintf("68%% = %.3g",  p68a), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
yline(ax1, -p68a, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
yline(ax1,  p99a, "k--", sprintf("99%% = %.3g",  p99a), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
yline(ax1, -p99a, "k--", "LineWidth", 1.4, "HandleVisibility", "off");
xlabel(ax1, "Predicted",  "FontSize", fontSize);
ylabel(ax1, "Residual",   "FontSize", fontSize);
title(ax1,  "Residuals vs predicted", "FontSize", fontSize);
grid(ax1, "on");

%% 2) Residual histogram
ax2 = nexttile(tl, 2);
histogram(ax2, res, 24, "EdgeColor", "k", "FaceAlpha", 0.75);
hold(ax2, "on");
xline(ax2, 0,    "r--", "LineWidth", 1.5);
xline(ax2,  p68a, "g--", sprintf("68%% = %.3g", p68a), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
xline(ax2, -p68a, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
xline(ax2,  p99a, "k--", sprintf("99%% = %.3g", p99a), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
xline(ax2, -p99a, "k--", "LineWidth", 1.4, "HandleVisibility", "off");
xlabel(ax2, "Residual", "FontSize", fontSize);
ylabel(ax2, "Count",    "FontSize", fontSize);
title(ax2, sprintf("Residual hist  (std=%.3g)", std(res)), "FontSize", fontSize);
grid(ax2, "on");

%% 3) QQ plot
ax3 = nexttile(tl, 3);
qqplot(ax3, res);
title(ax3, "QQ plot of residuals", "FontSize", fontSize);
grid(ax3, "on");

%% 4) Percentage residuals vs predicted
ax4 = nexttile(tl, 4);
scatter(ax4, yPred, percRes, 38, "filled", ...
    "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6);
hold(ax4, "on");
yline(ax4, 0,    "r--", "LineWidth", 1.5);
yline(ax4,  p68p, "g--", sprintf("68%% = %.2f%%",  p68p), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
yline(ax4, -p68p, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
yline(ax4,  p99p, "k--", sprintf("99%% = %.2f%%",  p99p), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
yline(ax4, -p99p, "k--", "LineWidth", 1.4, "HandleVisibility", "off");
xlabel(ax4, "Predicted",             "FontSize", fontSize);
ylabel(ax4, "Percentage error [%]",  "FontSize", fontSize);
title(ax4,  "% residuals vs predicted", "FontSize", fontSize);
grid(ax4, "on");

%% 5) Percentage residual histogram
ax5 = nexttile(tl, 5);
histogram(ax5, percRes, 24, "EdgeColor", "k", "FaceAlpha", 0.75);
hold(ax5, "on");
xline(ax5, 0,    "r--", "LineWidth", 1.5);
xline(ax5,  p68p, "g--", sprintf("68%% = %.2f%%", p68p), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
xline(ax5, -p68p, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
xline(ax5,  p99p, "k--", sprintf("99%% = %.2f%%", p99p), "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
xline(ax5, -p99p, "k--", "LineWidth", 1.4, "HandleVisibility", "off");
xlabel(ax5, "Percentage error [%]", "FontSize", fontSize);
ylabel(ax5, "Count",                "FontSize", fontSize);
title(ax5,  "% residual histogram", "FontSize", fontSize);
grid(ax5, "on");

%% 6) Percentage residuals vs true
ax6 = nexttile(tl, 6);
scatter(ax6, yTrue, percRes, 38, "filled", ...
    "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6);
hold(ax6, "on");
yline(ax6, 0, "r--", "LineWidth", 1.5);
xlabel(ax6, "True value",           "FontSize", fontSize);
ylabel(ax6, "Percentage error [%]", "FontSize", fontSize);
title(ax6,  "% residuals vs true",  "FontSize", fontSize);
grid(ax6, "on");

title(tl, sprintf("%s residual diagnostics | floor=%.3g", outputName, metrics.denomFloor), ...
    "FontSize", fontSize + 2, "Interpreter", "none");

exportFigure(fig, "Residuals_" + sanitizeFilename(outputName), expPath, expFmt);
end
