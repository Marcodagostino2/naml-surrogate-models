% function surrogatePlot(predictor, featureNames, trainSets, testSets, options)
% surrogatePlot  Plot aerodynamic surrogate diagnostics.
% 
% Generates parity plots, residual diagnostics, and physically meaningful 
% 1D/2D/3D slices of aerodynamic coefficients. It overlays the AeroDeck 
% grid against training anchors to visualize interpolation resolution and 
% surrogate smoothness, mitigating Runge's phenomenon.
% 
% INPUTS
%   predictor    : Function handle @(xUnit) -> prediction. Evaluates on the unit hypercube.
%   featureNames : String array (1 x d) of feature names (e.g., ["mach", "alpha", "phi", "hprot"]).
%   trainSets    : Struct array containing training anchors for each fidelity.
%   testSets     : (Optional) Struct array for testing (parity/residuals).
% 
% NAME-VALUE OPTIONS
%   outputName   : String specifying the aerodynamic coefficient (default: "Output").
%   ranges       : [2 x d] array of physical bounds for unit conversions.
%   power        : [1 x d] array of power exponents for unit conversions.
%   gridVectors  : Cell array of 1D vectors representing the AeroDeck nodes in the unit hypercube.
%   fontSize     : Base font size for plots (default: 11).
%   plotSurface  : Logical flag to enable/disable 1D/2D/3D physical plots (default: true).
%   exportPath   : Directory path to save plots (default: "").
%   exportFormat : Format for saved plots, "png" or "pdf" (default: "png").
% 
% arguments
%     predictor            (1,1) function_handle
%     featureNames         (1,:) string
%     trainSets            (1,:) 
%     testSets                    = []      
%     options.outputName   (1,1) string  = "Output"
%     options.ranges       (2,:) double  = []
%     options.power        (1,:) double  = []
%     options.gridVectors  (1,:) cell    = {}
%     options.fontSize     (1,1) double  = 11
%     options.plotSurface  (1,1) logical = true
%     options.exportPath   (1,1) string  = ""
%     options.exportFormat (1,1) string  = "png"
% end
% 
% % 1. PARITY & RESIDUALS
% numFidelities = length(trainSets);
% for iFid = 1:numFidelities
%     if ~isempty(testSets) && numel(testSets) >= iFid && size(testSets(iFid).xUnit, 1) > 0
%         yTestTrue = testSets(iFid).y;
%         yTestPred = predictor(testSets(iFid).xUnit);
% 
%         [rmse, r2, maxAE, res] = calculateMetrics(yTestTrue, yTestPred);
%         plotParity(yTestTrue, yTestPred, rmse, r2, maxAE, options.outputName, options.fontSize, options.exportPath, options.exportFormat);
%         plotResidualDiagnostics(yTestPred, res, options.outputName, options.fontSize, options.exportPath, options.exportFormat);
%     end
% end
% 
% if ~options.plotSurface || isempty(options.ranges)
%     return; % Physical plotting requires valid physical ranges
% end
% 
% % 2. PHYSICAL ROUTING
% Identify indices of key aerodynamic variables
% idxMach  = find(featureNames == "mach");
% idxAlpha = find(featureNames == "alpha");
% idxPhi   = find(featureNames == "phi");
% idxHprot = find(featureNames == "hprot");
% 
% Wrapper that accepts physical queries, converts to unit space, and calls the predictor
% physPredictor = @(xPhys) predictor(real2unit(xPhys, options.ranges, options.power));
% 
% fprintf("  Generating physical slices for %s...\n", options.outputName);
% 
% % 3. NOISE & GRID CHECK (Contour Map)
% Generate high-resolution 2D map (Mach vs Alpha) if dimensions are present
% if ~isempty(idxMach) && ~isempty(idxAlpha)
%     plotSmoothnessAndGrid(physPredictor, featureNames, trainSets, options.gridVectors, ...
%         idxMach, idxAlpha, options.ranges, options.power, options.outputName, options.fontSize);
% end
% 
% % 4. DOMAIN-SPECIFIC SLICES
% isAxial = startsWith(options.outputName, "CA", "IgnoreCase", true);
% isLateral = ismember(options.outputName, ["CY", "CN", "Cl", "Cm", "Cn"]);
% 
% if isAxial && ~isempty(idxMach)
%     plotAxialPhysics(physPredictor, featureNames, trainSets, idxMach, idxAlpha, idxHprot, idxPhi, ...
%         options.ranges, options.outputName, options.fontSize);
% 
% elseif isLateral && ~isempty(idxAlpha) && ~isempty(idxPhi)
%     plotLateralPhysics(physPredictor, featureNames, trainSets, idxMach, idxAlpha, idxPhi, ...
%         options.ranges, options.outputName, options.fontSize);
% end
% 
% end
% 
% % ========================================================================
%  PHYSICAL PLOTTING HELPERS
%  ========================================================================
% 
% function plotSmoothnessAndGrid(physPredictor, featNames, trainSets, gridVecs, idx1, idx2, ranges, power, outName, fontSize)
%     plotSmoothnessAndGrid Evaluates surrogate regularity and visualizes AeroDeck nodes.
%     figure("Name", "Smoothness & Grid Check", "NumberTitle", "off");
% 
%     Generate dense grid for derivative/smoothness evaluation
%     r1 = linspace(ranges(1, idx1), ranges(2, idx1), 100);
%     r2 = linspace(ranges(1, idx2), ranges(2, idx2), 100);
%     [R1, R2] = meshgrid(r1, r2);
% 
%     Base query evaluated at the dimensional mean
%     baseQuery = mean(ranges, 1);
%     Z = zeros(size(R1));
%     for i = 1:numel(R1)
%         q = baseQuery; 
%         q(idx1) = R1(i); 
%         q(idx2) = R2(i);
%         Z(i) = physPredictor(q);
%     end
% 
%     30-level contour map to highlight Kriging wiggles
%     contourf(R1, R2, Z, 30, 'LineStyle', 'none'); 
%     colormap turbo; colorbar; hold on;
% 
%     Overlay AeroDeck grid nodes (converted from unit back to physical space)
%     if ~isempty(gridVecs) && length(gridVecs) >= max(idx1, idx2)
%         Note: gridVecs are stored in unit coordinates. Convert them to physical.
%         physicalGrid1 = unit2real(gridVecs{idx1}(:), ranges(:, idx1), power(idx1));
%         physicalGrid2 = unit2real(gridVecs{idx2}(:), ranges(:, idx2), power(idx2));
% 
%         [G1, G2] = meshgrid(physicalGrid1, physicalGrid2);
%         scatter(G1(:), G2(:), 40, 'k+', 'LineWidth', 1.2, 'DisplayName', 'AeroDeck Nodes');
%     end
% 
%     Overlay actual training anchors
%     trainX1 = trainSets(end).xReal(:, idx1);
%     trainX2 = trainSets(end).xReal(:, idx2);
%     scatter(trainX1, trainX2, 20, 'wo', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Training Anchors');
% 
%     xlabel(featNames(idx1), "FontSize", fontSize);
%     ylabel(featNames(idx2), "FontSize", fontSize);
%     title(sprintf("%s Smoothness Check\n(Color = Pred, + = LookUp Grid, o = Train)", outName), "FontSize", fontSize);
%     legend('Location', 'best'); grid on;
% end
% 
% function plotAxialPhysics(physPredictor, ~, ~, idxMach, idxAlpha, idxHprot, idxPhi, ranges, outName, fontSize)
%     plotAxialPhysics Plots 1D axial force physics (CA) vs Mach, parameterized by hprot and Alpha.
%     figure("Name", "Axial Force Physics", "NumberTitle", "off", "Position", [100 100 1200 400]);
% 
%     alphas = [0, 5, 10]; 
%     hprots = [];
%     if ~isempty(idxHprot), hprots = [0, 50, 100]; end
%     if isempty(hprots), hprots = 0; end
% 
%     machLine = linspace(ranges(1, idxMach), ranges(2, idxMach), 50)';
%     colors = lines(length(hprots));
% 
%     for a = 1:length(alphas)
%         subplot(1, length(alphas), a); hold on;
% 
%         for h = 1:length(hprots)
%             baseQuery = mean(ranges, 1);
%             if ~isempty(idxAlpha), baseQuery(idxAlpha) = alphas(a); end
%             if ~isempty(idxPhi),   baseQuery(idxPhi)   = 0; end
% 
%             yLine = zeros(size(machLine));
%             for i = 1:length(machLine)
%                 q = baseQuery; 
%                 q(idxMach) = machLine(i);
%                 if ~isempty(idxHprot), q(idxHprot) = hprots(h); end
%                 yLine(i) = physPredictor(q);
%             end
% 
%             plot(machLine, yLine, 'LineWidth', 2, 'Color', colors(h,:), 'DisplayName', sprintf('hprot = %d%%', hprots(h)));
%         end
%         xlabel("Mach", "FontSize", fontSize);
%         ylabel(outName, "FontSize", fontSize);
%         title(sprintf("Alpha = %d deg", alphas(a)), "FontSize", fontSize);
%         grid on; 
%         if a == 1, legend('Location', 'best'); end
%     end
%     sgtitle(sprintf("Axial Physics: %s vs Mach", outName), "FontSize", fontSize + 2);
% end
% 
% function plotLateralPhysics(physPredictor, ~, ~, idxMach, idxAlpha, idxPhi, ranges, outName, fontSize)
%     plotLateralPhysics Plots 3D lateral force physics (e.g., CN, CY) vs Alpha and Phi.
%     figure("Name", "Lateral Force Physics", "NumberTitle", "off");
% 
%     alphaLine = linspace(ranges(1, idxAlpha), ranges(2, idxAlpha), 30);
%     phiLine   = linspace(ranges(1, idxPhi), ranges(2, idxPhi), 30);
%     [A, P] = meshgrid(alphaLine, phiLine);
% 
%     baseQuery = mean(ranges, 1);
%     if ~isempty(idxMach) 
%         Evaluate at Mach 1.5 if within valid bounds, else at max available
%         baseQuery(idxMach) = min(1.5, ranges(2, idxMach)); 
%     end 
% 
%     Z = zeros(size(A));
%     for i = 1:numel(A)
%         q = baseQuery; 
%         q(idxAlpha) = A(i); 
%         q(idxPhi) = P(i);
%         Z(i) = physPredictor(q);
%     end
% 
%     surf(A, P, Z, 'FaceAlpha', 0.8, 'EdgeColor', 'interp'); 
%     colormap parula; hold on;
% 
%     xlabel("Alpha [deg]", "FontSize", fontSize);
%     ylabel("Phi [deg]", "FontSize", fontSize);
%     zlabel(outName, "FontSize", fontSize);
% 
%     machVal = "N/A";
%     if ~isempty(idxMach), machVal = sprintf('%.2f', baseQuery(idxMach)); end
%     title(sprintf("%s vs Alpha & Phi\n(Evaluated at Mach = %s)", outName, machVal), "FontSize", fontSize);
%     view(-45, 30); grid on;
% end
% 
% % ========================================================================
%  METRICS & GENERIC HELPERS
%  ========================================================================
% 
% function [rmse, r2, maxAE, residuals] = calculateMetrics(yTrue, yPred)
%     residuals = yTrue - yPred;
%     rmse  = sqrt(mean(residuals.^2));
%     r2    = 1 - sum(residuals.^2) / sum((yTrue - mean(yTrue)).^2);
%     maxAE = max(abs(residuals));
% end
% 
% % -------- PLOTTING HELPERS --------
% function exportFigure(fig, baseName, expPath, expFmt)
%     if strlength(expPath) > 0
%         if ~exist(expPath, 'dir'), mkdir(expPath); end
%         fileName = fullfile(expPath, sprintf("%s.%s", baseName, lower(expFmt)));
%         if lower(expFmt) == "pdf"
%             exportgraphics(fig, fileName, 'ContentType', 'vector');
%         else
%             exportgraphics(fig, fileName, 'Resolution', 300);
%         end
%     end
% end
% 
% function plotParity(yTrue, yPred, rmse, r2, maxAE, outputName, fontSize, expPath, expFmt)
%     fig = figure("Name", "Surrogate Parity", "NumberTitle", "off");
%     scatter(yTrue, yPred, 50, "filled", "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6); hold on;
%     axisLimits = [min([yTrue; yPred]), max([yTrue; yPred])];
%     plot(axisLimits, axisLimits, "r--", "LineWidth", 2);
%     xlabel("True " + outputName, "FontSize", fontSize);
%     ylabel("Predicted " + outputName, "FontSize", fontSize);
%     title(sprintf("Parity (%s)\nRMSE: %.4g | R^2: %.4f | MaxAE: %.4g", outputName, rmse, r2, maxAE), "FontSize", fontSize + 2);
%     grid on; axis equal;
% 
%     exportFigure(fig, "Parity_" + outputName, expPath, expFmt);
% end
% 
% function plotResidualDiagnostics(yPred, residuals, outputName, fontSize, expPath, expFmt)
%     absRes = abs(residuals); 
%     perc68 = prctile(absRes, 68); 
%     perc99 = prctile(absRes, 99);
% 
%     fig = figure("Name", "Residuals", "NumberTitle", "off", 'Position', [100, 100, 1200, 400]);
% 
%     subplot(1,3,1);
%     scatter(yPred, residuals, 50, "filled", "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6); hold on;
%     plot([min(yPred) max(yPred)], [0 0], "r--", "LineWidth", 2);
%     yline( perc68, "g--", sprintf('68%% (%.3g)',  perc68), "LineWidth", 2, "LabelHorizontalAlignment", "left", "LabelVerticalAlignment", "bottom", "FontSize", fontSize-2);
%     yline(-perc68, "g--", sprintf('68%% (%.3g)', -perc68), "LineWidth", 2, "LabelHorizontalAlignment", "left", "LabelVerticalAlignment", "top", "FontSize", fontSize-2);
%     yline( perc99, "k--", sprintf('99%% (%.3g)',  perc99), "LineWidth", 2, "LabelHorizontalAlignment", "left", "LabelVerticalAlignment", "bottom", "FontSize", fontSize-2);
%     yline(-perc99, "k--", sprintf('99%% (%.3g)', -perc99), "LineWidth", 2, "LabelHorizontalAlignment", "left", "LabelVerticalAlignment", "top", "FontSize", fontSize-2);
%     xlabel("Predicted " + outputName, "FontSize", fontSize); 
%     ylabel("Residual", "FontSize", fontSize);
%     title("Residuals vs Prediction", "FontSize", fontSize); grid on;
% 
%     subplot(1,3,2);
%     histogram(residuals, 20, "EdgeColor", "k", "FaceAlpha", 0.7); hold on;
%     xline( perc68, "g--", sprintf('68%% (%.3g)',  perc68), "LineWidth", 2, "LabelVerticalAlignment", "top", "LabelHorizontalAlignment", "left", "FontSize", fontSize-2);
%     xline(-perc68, "g--", sprintf('68%% (%.3g)', -perc68), "LineWidth", 2, "LabelVerticalAlignment", "top", "LabelHorizontalAlignment", "right", "FontSize", fontSize-2);
%     xline( perc99, "k--", sprintf('99%% (%.3g)',  perc99), "LineWidth", 2, "LabelVerticalAlignment", "bottom", "LabelHorizontalAlignment", "left", "FontSize", fontSize-2);
%     xline(-perc99, "k--", sprintf('99%% (%.3g)', -perc99), "LineWidth", 2, "LabelVerticalAlignment", "bottom", "LabelHorizontalAlignment", "right", "FontSize", fontSize-2);
%     xlabel("Residual", "FontSize", fontSize); 
%     ylabel("Frequency", "FontSize", fontSize);
%     title(sprintf("Histogram (Std = %.3g)", std(residuals)), "FontSize", fontSize); grid on;
% 
%     subplot(1,3,3); 
%     qqplot(residuals); 
%     title("Q-Q Plot", "FontSize", fontSize); grid on;
% 
%     sgtitle("Residual Diagnostics: " + outputName, "FontSize", fontSize + 2);
%     exportFigure(fig, "Residuals_" + outputName, expPath, expFmt);
% end


function surrogatePlot(predictor, featureNames, trainSets, testSets, options)
% surrogatePlot  Plot diagnostics for a generic aerodynamic surrogate model.
%
% This utility generates:
%   1) parity plots on available test sets,
%   2) residual diagnostics using both absolute and percentage errors,
%   3) physical-space slices and contour diagnostics when physical ranges
%      are available.
%
% The plotting workflow is generic and can be used for Kriging or
% Co-Kriging surrogates, provided that:
%   - predictor accepts unit-space inputs,
%   - trainSets/testSets contain xUnit and/or xReal,
%   - real2unit/unit2real are available in the project.
%
% INPUTS
%   predictor    : Function handle @(xUnit) -> prediction.
%                  It must evaluate the model in unit space.
%   featureNames : String array (1 x d) of feature names
%                  (e.g. ["mach","alpha","phi","hprot"]).
%   trainSets    : Struct array containing training data for each fidelity.
%                  Recommended fields: xUnit, xReal, y.
%   testSets     : Optional struct array containing test data.
%
% NAME-VALUE OPTIONS
%   outputName   : Output label used in plot titles and file names.
%   ranges       : [2 x d] physical bounds for unit conversion.
%   power        : [1 x d] power exponents for unit conversion.
%   gridVectors  : Cell array of AeroDeck nodes in unit space.
%   fontSize     : Base font size for plots.
%   plotSurface  : Enable physical-space plots.
%   exportPath   : Export folder. If empty, plots are not saved.
%   exportFormat : "png" or "pdf".
%   relErrFloor  : Relative denominator floor, expressed as a fraction of
%                  max(abs(yTrue)), used to avoid exploding percentage
%                  errors when the true value is near zero.
%
% NOTES
%   - Percentage errors are "safeguarded" near zero using relErrFloor.
%   - For aerodynamic coefficients crossing zero, percentage metrics should
%     be interpreted with care.

arguments
    predictor            (1,1) function_handle
    featureNames         (1,:) string
    trainSets            (1,:)
    testSets                    = []
    options.outputName   (1,1) string  = "Output"
    options.ranges       (2,:) double  = []
    options.power        (1,:) double  = []
    options.gridVectors  (1,:) cell    = {}
    options.fontSize     (1,1) double  = 11
    options.plotSurface  (1,1) logical = true
    options.exportPath   (1,1) string  = ""
    options.exportFormat (1,1) string  = "png"
    options.relErrFloor  (1,1) double  = 0.02
end

%% 1) PARITY AND RESIDUAL DIAGNOSTICS
numFidelities = numel(trainSets);

for iFid = 1:numFidelities
    hasTest = ~isempty(testSets) && numel(testSets) >= iFid ...
        && isfield(testSets(iFid), "xUnit") ...
        && ~isempty(testSets(iFid).xUnit) ...
        && isfield(testSets(iFid), "y") ...
        && ~isempty(testSets(iFid).y);

    if ~hasTest
        continue
    end

    yTestTrue = testSets(iFid).y(:);
    yTestPred = predictor(testSets(iFid).xUnit);
    yTestPred = yTestPred(:);

    metrics = calculateMetrics(yTestTrue, yTestPred, options.relErrFloor);
    suffix  = makeFidelitySuffix(iFid, numFidelities);

    plotParity( ...
        yTestTrue, yTestPred, metrics, ...
        options.outputName + suffix, options.fontSize, ...
        options.exportPath, options.exportFormat);

    plotResidualDiagnostics( ...
        yTestTrue, yTestPred, metrics, ...
        options.outputName + suffix, options.fontSize, ...
        options.exportPath, options.exportFormat);
end

if ~options.plotSurface || isempty(options.ranges)
    return
end

%% 2) PHYSICAL ROUTING
idxMach  = find(featureNames == "mach",  1);
idxAlpha = find(featureNames == "alpha", 1);
idxPhi   = find(featureNames == "phi",   1);
idxHprot = find(featureNames == "hprot", 1);

if isempty(options.power)
    options.power = ones(1, size(options.ranges, 2));
end

physPredictor = @(xPhys) predictor(real2unit(xPhys, options.ranges, options.power));

fprintf("  Generating physical slices for %s...\n", options.outputName);

%% 3) GENERIC SMOOTHNESS CHECK
if ~isempty(idxMach) && ~isempty(idxAlpha)
    plotSmoothnessAndGrid( ...
        physPredictor, featureNames, trainSets, options.gridVectors, ...
        idxMach, idxAlpha, options.ranges, options.power, ...
        options.outputName, options.fontSize, ...
        options.exportPath, options.exportFormat);
end

%% 4) DOMAIN-SPECIFIC SLICES
isAxial   = startsWith(options.outputName, "CA", "IgnoreCase", true);
isLateral = ismember(options.outputName, ["CY", "CN", "Cl", "Cm", "Cn"]);

if isAxial && ~isempty(idxMach)
    plotAxialPhysics( ...
        physPredictor, featureNames, trainSets, ...
        idxMach, idxAlpha, idxHprot, idxPhi, ...
        options.ranges, options.outputName, options.fontSize, ...
        options.exportPath, options.exportFormat);

elseif isLateral && ~isempty(idxAlpha) && ~isempty(idxPhi)
    plotLateralPhysics( ...
        physPredictor, featureNames, trainSets, ...
        idxMach, idxAlpha, idxPhi, ...
        options.ranges, options.outputName, options.fontSize, ...
        options.exportPath, options.exportFormat);
end

end

%% ========================================================================
%  PHYSICAL PLOTTING HELPERS
%  ========================================================================
function plotSmoothnessAndGrid(physPredictor, featNames, trainSets, gridVecs, ...
    idx1, idx2, ranges, power, outName, fontSize, expPath, expFmt)

    fig = figure("Name", "Smoothness and Grid Check", ...
        "NumberTitle", "off");

    r1 = linspace(ranges(1, idx1), ranges(2, idx1), 100);
    r2 = linspace(ranges(1, idx2), ranges(2, idx2), 100);
    [R1, R2] = meshgrid(r1, r2);

    baseQuery = mean(ranges, 1);
    Z = zeros(size(R1));

    for i = 1:numel(R1)
        q = baseQuery;
        q(idx1) = R1(i);
        q(idx2) = R2(i);
        Z(i) = physPredictor(q);
    end

    contourf(R1, R2, Z, 30, "LineStyle", "none");
    hold on
    contour(R1, R2, Z, 12, "k-", "LineWidth", 0.35, ...
        "HandleVisibility", "off");
    colormap turbo
    cb = colorbar;
    cb.Label.String = outName;

    if ~isempty(gridVecs) && numel(gridVecs) >= max(idx1, idx2) ...
            && ~isempty(gridVecs{idx1}) && ~isempty(gridVecs{idx2})

        physicalGrid1 = unit2real(gridVecs{idx1}(:), ranges(:, idx1), power(idx1));
        physicalGrid2 = unit2real(gridVecs{idx2}(:), ranges(:, idx2), power(idx2));
        [G1, G2] = meshgrid(physicalGrid1, physicalGrid2);

        scatter(G1(:), G2(:), 36, "k+", "LineWidth", 1.0, ...
            "DisplayName", "AeroDeck nodes");
    end

    if isfield(trainSets(end), "xReal") && ~isempty(trainSets(end).xReal)
        trainX1 = trainSets(end).xReal(:, idx1);
        trainX2 = trainSets(end).xReal(:, idx2);
        scatter(trainX1, trainX2, 22, "wo", "filled", ...
            "MarkerEdgeColor", "k", "DisplayName", "Training points");
    end

    xlabel(featNames(idx1), "FontSize", fontSize, "Interpreter", "none");
    ylabel(featNames(idx2), "FontSize", fontSize, "Interpreter", "none");
    title(sprintf("%s smoothness map", outName), ...
        "FontSize", fontSize + 1, "Interpreter", "none");
    legend("Location", "best")
    grid on

    exportFigure(fig, "Smoothness_" + sanitizeFilename(outName), expPath, expFmt);
end

function plotAxialPhysics(physPredictor, ~, ~, idxMach, idxAlpha, idxHprot, idxPhi, ...
    ranges, outName, fontSize, expPath, expFmt)

    fig = figure("Name", "Axial Physics", "NumberTitle", "off");

    alphas = [0, 5, 10];
    if ~isempty(idxAlpha)
        alphas = alphas(alphas >= ranges(1, idxAlpha) & alphas <= ranges(2, idxAlpha));
        if isempty(alphas)
            alphas = linspace(ranges(1, idxAlpha), ranges(2, idxAlpha), 3);
        end
    else
        alphas = 0;
    end

    hprots = [];
    if ~isempty(idxHprot)
        hprots = [0, 50, 100];
        hprots = hprots(hprots >= ranges(1, idxHprot) & hprots <= ranges(2, idxHprot));
        if isempty(hprots)
            hprots = linspace(ranges(1, idxHprot), ranges(2, idxHprot), 3);
        end
    end
    if isempty(hprots)
        hprots = 0;
    end

    machLine = linspace(ranges(1, idxMach), ranges(2, idxMach), 80)';
    colors = lines(numel(hprots));

    tl = tiledlayout(fig, 1, numel(alphas), "TileSpacing", "compact", "Padding", "compact");

    for a = 1:numel(alphas)
        ax = nexttile(tl, a);
        hold(ax, "on")

        for h = 1:numel(hprots)
            baseQuery = mean(ranges, 1);

            if ~isempty(idxAlpha)
                baseQuery(idxAlpha) = alphas(a);
            end
            if ~isempty(idxPhi)
                baseQuery(idxPhi) = clampToBounds(0, ranges(:, idxPhi));
            end

            yLine = zeros(size(machLine));
            for i = 1:numel(machLine)
                q = baseQuery;
                q(idxMach) = machLine(i);
                if ~isempty(idxHprot)
                    q(idxHprot) = hprots(h);
                end
                yLine(i) = physPredictor(q);
            end

            plot(ax, machLine, yLine, "LineWidth", 2, ...
                "Color", colors(h,:), ...
                "DisplayName", sprintf("hprot = %.0f%%", hprots(h)));
        end

        xlabel(ax, "Mach", "FontSize", fontSize);
        ylabel(ax, outName, "FontSize", fontSize, "Interpreter", "none");
        title(ax, sprintf("alpha = %.2f deg", alphas(a)), ...
            "FontSize", fontSize, "Interpreter", "none");
        grid(ax, "on")

        if a == 1
            legend(ax, "Location", "best")
        end
    end

    title(tl, sprintf("Axial physics: %s versus Mach", outName), ...
        "FontSize", fontSize + 2, "Interpreter", "none");

    exportFigure(fig, "AxialPhysics_" + sanitizeFilename(outName), expPath, expFmt);
end

function plotLateralPhysics(physPredictor, ~, ~, idxMach, idxAlpha, idxPhi, ...
    ranges, outName, fontSize, expPath, expFmt)

    fig = figure("Name", "Lateral Physics", "NumberTitle", "off");

    alphaLine = linspace(ranges(1, idxAlpha), ranges(2, idxAlpha), 60);
    phiLine   = linspace(ranges(1, idxPhi),   ranges(2, idxPhi),   60);
    [A, P] = meshgrid(alphaLine, phiLine);

    baseQuery = mean(ranges, 1);
    if ~isempty(idxMach)
        baseQuery(idxMach) = min(max(1.5, ranges(1, idxMach)), ranges(2, idxMach));
    end

    Z = zeros(size(A));
    for i = 1:numel(A)
        q = baseQuery;
        q(idxAlpha) = A(i);
        q(idxPhi)   = P(i);
        Z(i) = physPredictor(q);
    end

    contourf(A, P, Z, 30, "LineStyle", "none");
    hold on
    contour(A, P, Z, 12, "k-", "LineWidth", 0.35, "HandleVisibility", "off");
    colormap parula
    cb = colorbar;
    cb.Label.String = outName;

    xlabel("alpha [deg]", "FontSize", fontSize);
    ylabel("phi [deg]",   "FontSize", fontSize);
    grid on

    machVal = "N/A";
    if ~isempty(idxMach)
        machVal = sprintf("%.2f", baseQuery(idxMach));
    end

    title(sprintf("%s in the alpha-phi plane (Mach = %s)", outName, machVal), ...
        "FontSize", fontSize + 1, "Interpreter", "none");

    exportFigure(fig, "LateralPhysics_" + sanitizeFilename(outName), expPath, expFmt);
end

%% ========================================================================
%  METRICS
%  ========================================================================
function metrics = calculateMetrics(yTrue, yPred, relErrFloor)

    yTrue = yTrue(:);
    yPred = yPred(:);

    residuals    = yTrue - yPred;
    absResiduals = abs(residuals);

    rmse = sqrt(mean(residuals.^2));

    denomR2 = sum((yTrue - mean(yTrue)).^2);
    if denomR2 < eps
        r2 = NaN;
    else
        r2 = 1 - sum(residuals.^2) / denomR2;
    end

    maxAE = max(absResiduals);
    mae   = mean(absResiduals);

    scaleRef = max(abs(yTrue));
    denomFloor = max(eps, relErrFloor * max(scaleRef, 1));
    relDenom = max(abs(yTrue), denomFloor);

    percResiduals = 100 * residuals ./ relDenom;
    absPercError  = abs(percResiduals);

    mape   = mean(absPercError);
    maxAPE = max(absPercError);

    metrics.rmse          = rmse;
    metrics.r2            = r2;
    metrics.maxAE         = maxAE;
    metrics.mae           = mae;
    metrics.residuals     = residuals;
    metrics.absResiduals  = absResiduals;
    metrics.percResiduals = percResiduals;
    metrics.mape          = mape;
    metrics.maxAPE        = maxAPE;
    metrics.denomFloor    = denomFloor;
end

%% ========================================================================
%  GENERIC HELPERS
%  ========================================================================
function x = clampToBounds(x, bounds)
    x = min(max(x, bounds(1)), bounds(2));
end

function suffix = makeFidelitySuffix(iFid, numFidelities)
    if numFidelities <= 1
        suffix = "";
    else
        suffix = sprintf("_Fid%d", iFid);
    end
end

function nameOut = sanitizeFilename(nameIn)
    nameOut = string(regexprep(char(nameIn), '[^\w\-]', '_'));
end

%% ========================================================================
%  EXPORT AND GENERIC PLOTS
%  ========================================================================
function exportFigure(fig, baseName, expPath, expFmt)
    if strlength(expPath) == 0
        return
    end

    if ~exist(expPath, "dir")
        mkdir(expPath);
    end

    fileName = fullfile(expPath, sprintf("%s.%s", baseName, lower(expFmt)));

    if lower(expFmt) == "pdf"
        exportgraphics(fig, fileName, "ContentType", "vector");
    else
        exportgraphics(fig, fileName, "Resolution", 300);
    end
end

function plotParity(yTrue, yPred, metrics, outputName, fontSize, expPath, expFmt)

    fig = figure("Name", "Parity Plot", "NumberTitle", "off");

    tl = tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");

    residuals    = metrics.residuals;
    absResiduals = metrics.absResiduals;
    percRes      = metrics.percResiduals;

    lo = min([yTrue; yPred]);
    hi = max([yTrue; yPred]);
    if abs(hi - lo) < eps
        lo = lo - 1;
        hi = hi + 1;
    end

    % --- 1) Standard parity
    ax1 = nexttile(tl, 1);
    scatter(ax1, yTrue, yPred, 42, "filled", ...
        "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.65);
    hold(ax1, "on")
    plot(ax1, [lo hi], [lo hi], "r--", "LineWidth", 2)

    xlabel(ax1, "True value", "FontSize", fontSize);
    ylabel(ax1, "Predicted value", "FontSize", fontSize);
    title(ax1, sprintf("%s parity", outputName), ...
        "FontSize", fontSize + 1, "Interpreter", "none");
    grid(ax1, "on")
    axis(ax1, "equal")
    xlim(ax1, [lo hi])
    ylim(ax1, [lo hi])

    % --- 2) Absolute error versus true value
    ax2 = nexttile(tl, 2);
    scatter(ax2, yTrue, absResiduals, 42, "filled", ...
        "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.65);
    hold(ax2, "on")
    yline(ax2, metrics.mae,   "b--", sprintf("MAE = %.3g", metrics.mae), ...
        "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
    yline(ax2, metrics.maxAE, "r--", sprintf("MaxAE = %.3g", metrics.maxAE), ...
        "LineWidth", 1.6, "LabelHorizontalAlignment", "left");

    xlabel(ax2, "True value", "FontSize", fontSize);
    ylabel(ax2, "Absolute error", "FontSize", fontSize);
    title(ax2, "Absolute error", "FontSize", fontSize);
    grid(ax2, "on")

    % --- 3) Percentage error versus true value
    ax3 = nexttile(tl, 3);
    scatter(ax3, yTrue, percRes, 42, "filled", ...
        "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.65);
    hold(ax3, "on")
    yline(ax3, 0, "k-", "LineWidth", 1.0, "HandleVisibility", "off");
    yline(ax3,  metrics.mape, "g--", sprintf("MAPE = %.2f%%", metrics.mape), ...
        "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
    yline(ax3, -metrics.mape, "g--", "LineWidth", 1.6, "HandleVisibility", "off");

    xlabel(ax3, "True value", "FontSize", fontSize);
    ylabel(ax3, "Percentage error [%]", "FontSize", fontSize);
    title(ax3, "Signed percentage error", "FontSize", fontSize);
    grid(ax3, "on")

    % --- 4) Percentage error histogram
    ax4 = nexttile(tl, 4);
    histogram(ax4, percRes, 25, "EdgeColor", "k", "FaceAlpha", 0.75);
    hold(ax4, "on")
    xline(ax4, 0, "k-", "LineWidth", 1.0, "HandleVisibility", "off");
    xline(ax4,  metrics.mape, "g--", sprintf("MAPE = %.2f%%", metrics.mape), ...
        "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
    xline(ax4, -metrics.mape, "g--", "LineWidth", 1.6, "HandleVisibility", "off");
    xline(ax4,  metrics.maxAPE, "r--", sprintf("MaxAPE = %.2f%%", metrics.maxAPE), ...
        "LineWidth", 1.6, "LabelHorizontalAlignment", "left");
    xline(ax4, -metrics.maxAPE, "r--", "LineWidth", 1.6, "HandleVisibility", "off");

    xlabel(ax4, "Percentage error [%]", "FontSize", fontSize);
    ylabel(ax4, "Count", "FontSize", fontSize);
    title(ax4, "Percentage error histogram", "FontSize", fontSize);
    grid(ax4, "on")

    title(tl, sprintf("%s | RMSE = %.4g | R^2 = %.4f | MaxAE = %.4g | MAPE = %.2f%%", ...
        outputName, metrics.rmse, metrics.r2, metrics.maxAE, metrics.mape), ...
        "FontSize", fontSize + 2, "Interpreter", "none");

    exportFigure(fig, "Parity_" + sanitizeFilename(outputName), expPath, expFmt);
end

function plotResidualDiagnostics(yTrue, yPred, metrics, outputName, fontSize, expPath, expFmt)

    residuals    = metrics.residuals;
    absResiduals = metrics.absResiduals;
    percRes      = metrics.percResiduals;
    absPercRes   = abs(percRes);

    perc68Abs = prctile(absResiduals, 68);
    perc99Abs = prctile(absResiduals, 99);
    perc68Pct = prctile(absPercRes, 68);
    perc99Pct = prctile(absPercRes, 99);

    fig = figure("Name", "Residual Diagnostics", "NumberTitle", "off");

    tl = tiledlayout(fig, 2, 3, "TileSpacing", "compact", "Padding", "compact");

    % --- 1) Absolute residuals versus prediction
    ax1 = nexttile(tl, 1);
    scatter(ax1, yPred, residuals, 38, "filled", ...
        "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6);
    hold(ax1, "on")
    yline(ax1, 0, "r--", "LineWidth", 1.5);
    yline(ax1,  perc68Abs, "g--", sprintf("68%% = %.3g",  perc68Abs), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    yline(ax1, -perc68Abs, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
    yline(ax1,  perc99Abs, "k--", sprintf("99%% = %.3g",  perc99Abs), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    yline(ax1, -perc99Abs, "k--", "LineWidth", 1.4, "HandleVisibility", "off");

    xlabel(ax1, "Predicted value", "FontSize", fontSize);
    ylabel(ax1, "Residual", "FontSize", fontSize);
    title(ax1, "Residuals versus prediction", "FontSize", fontSize);
    grid(ax1, "on")

    % --- 2) Absolute residual histogram
    ax2 = nexttile(tl, 2);
    histogram(ax2, residuals, 24, "EdgeColor", "k", "FaceAlpha", 0.75);
    hold(ax2, "on")
    xline(ax2, 0, "r--", "LineWidth", 1.5);
    xline(ax2,  perc68Abs, "g--", sprintf("68%% = %.3g", perc68Abs), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    xline(ax2, -perc68Abs, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
    xline(ax2,  perc99Abs, "k--", sprintf("99%% = %.3g", perc99Abs), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    xline(ax2, -perc99Abs, "k--", "LineWidth", 1.4, "HandleVisibility", "off");

    xlabel(ax2, "Residual", "FontSize", fontSize);
    ylabel(ax2, "Count", "FontSize", fontSize);
    title(ax2, sprintf("Residual histogram (std = %.3g)", std(residuals)), ...
        "FontSize", fontSize);
    grid(ax2, "on")

    % --- 3) QQ plot for absolute residuals
    ax3 = nexttile(tl, 3);
    qqplot(ax3, residuals);
    title(ax3, "QQ plot of residuals", "FontSize", fontSize);
    grid(ax3, "on")

    % --- 4) Percentage residuals versus prediction
    ax4 = nexttile(tl, 4);
    scatter(ax4, yPred, percRes, 38, "filled", ...
        "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6);
    hold(ax4, "on")
    yline(ax4, 0, "r--", "LineWidth", 1.5);
    yline(ax4,  perc68Pct, "g--", sprintf("68%% = %.2f%%",  perc68Pct), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    yline(ax4, -perc68Pct, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
    yline(ax4,  perc99Pct, "k--", sprintf("99%% = %.2f%%",  perc99Pct), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    yline(ax4, -perc99Pct, "k--", "LineWidth", 1.4, "HandleVisibility", "off");

    xlabel(ax4, "Predicted value", "FontSize", fontSize);
    ylabel(ax4, "Percentage residual [%]", "FontSize", fontSize);
    title(ax4, "Percentage residuals versus prediction", "FontSize", fontSize);
    grid(ax4, "on")

    % --- 5) Percentage residual histogram
    ax5 = nexttile(tl, 5);
    histogram(ax5, percRes, 24, "EdgeColor", "k", "FaceAlpha", 0.75);
    hold(ax5, "on")
    xline(ax5, 0, "r--", "LineWidth", 1.5);
    xline(ax5,  perc68Pct, "g--", sprintf("68%% = %.2f%%", perc68Pct), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    xline(ax5, -perc68Pct, "g--", "LineWidth", 1.4, "HandleVisibility", "off");
    xline(ax5,  perc99Pct, "k--", sprintf("99%% = %.2f%%", perc99Pct), ...
        "LineWidth", 1.4, "LabelHorizontalAlignment", "left");
    xline(ax5, -perc99Pct, "k--", "LineWidth", 1.4, "HandleVisibility", "off");

    xlabel(ax5, "Percentage residual [%]", "FontSize", fontSize);
    ylabel(ax5, "Count", "FontSize", fontSize);
    title(ax5, "Percentage residual histogram", "FontSize", fontSize);
    grid(ax5, "on")

    % --- 6) Percentage residuals versus true value
    ax6 = nexttile(tl, 6);
    scatter(ax6, yTrue, percRes, 38, "filled", ...
        "MarkerEdgeColor", "k", "MarkerFaceAlpha", 0.6);
    hold(ax6, "on")
    yline(ax6, 0, "r--", "LineWidth", 1.5);

    xlabel(ax6, "True value", "FontSize", fontSize);
    ylabel(ax6, "Percentage residual [%]", "FontSize", fontSize);
    title(ax6, "Percentage residuals versus true value", "FontSize", fontSize);
    grid(ax6, "on")

    title(tl, sprintf("%s residual diagnostics | denominator floor = %.3g", ...
        outputName, metrics.denomFloor), ...
        "FontSize", fontSize + 2, "Interpreter", "none");

    exportFigure(fig, "Residuals_" + sanitizeFilename(outputName), expPath, expFmt);
end
