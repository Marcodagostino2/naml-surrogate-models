classdef SurrogatePlotter < handle
% SurrogatePlotter  Diagnostics and parametric surface plots for SurrogateModel.
%
% Generates publication-ready 1D slices, 2D contours, 3D surfaces, and parity plots
% using strictly physical units, LaTeX formatting, and multi-fidelity data overlays.
%
% USAGE:
%   p = SurrogatePlotter(model, trainSets, testSet, metadata, "units", datasetUnits);
%   p.plotResiduals(1); % Parity plot
%   p.plot1DSlices(1);  % 1D parametric variations
%   p.plot3DSurfaces(1);% 3D physical surfaces

    properties (Access = private)
        predictFcn   function_handle
        varianceFcn  function_handle
        trainSets    (1,:) struct   % Array of structs (LF, HF)
        testSet      (1,1) struct   % Highest fidelity test set
        
        inNames      (1,:) string
        outNames     (1,:) string
        ranges       (2,:) double
        power        (1,:) double
        hyperparams  struct
        units        (1,1) struct
        
        % Default Operating Point (Slice centers)
        opUnit       (1,:) double
        opReal       (1,:) double
        
        % Styling & Export Options
        maxDisplayPts (1,1) double = 300
        exportPath    (1,1) string = ""
        exportFormat  (1,1) string = "png"
        fontSize      (1,1) double = 11
        
        % Multi-fidelity Colors
        cLfTrain = [0.45 0.55 0.70] % Muted Blue for Low Fidelity
        cHfTrain = [0.85 0.25 0.15] % Crimson for High Fidelity
        cTest    = [0.20 0.60 0.30] % Forest Green for Test
    end
    
    methods
        function obj = SurrogatePlotter(model, trainSets, testSet, metadata, options)
            arguments
                model
                trainSets    (1,:) struct
                testSet      (1,1) struct
                metadata     struct
                options.units         (1,1) struct = struct()
                options.maxDisplayPts (1,1) double = 300
                options.exportPath    (1,1) string = ""
                options.exportFormat  (1,1) string = "png"
                options.fontSize      (1,1) double = 11
            end
            
            obj.predictFcn  = @(x) model.predict(x);
            obj.varianceFcn = @(x) obj.safeVariance(model, x);
            obj.trainSets   = trainSets;
            obj.testSet     = testSet;
            
            obj.inNames     = string(metadata.inNames);
            obj.outNames    = string(metadata.outNames);
            obj.ranges      = metadata.ranges;
            obj.power       = metadata.power;
            obj.hyperparams = model.hyperparams;
            obj.units       = options.units;
            
            obj.maxDisplayPts = options.maxDisplayPts;
            obj.exportPath    = options.exportPath;
            obj.exportFormat  = options.exportFormat;
            obj.fontSize      = options.fontSize;
            
            % Setup Default Operating Point (Middle of physical bounds or provided)
            obj.setupOperatingPoint(metadata);
        end
        
        function plotResiduals(obj, outputIdx)
            % plotResiduals  Parity and error distributions for the test set.
            % Uses raw percentage error (no relErrFloor thresholding).
            arguments
                obj
                outputIdx (1,1) double = 1
            end
            
            yTrue = obj.testSet.y(:, outputIdx);
            yFull = obj.predictFcn(obj.testSet.xUnit);
            yPred = yFull(:, outputIdx);
            
            % calculateMetrics called with 0 relFloor to keep raw relative errors
            m = calculateMetrics(yTrue, yPred, 0); 
            name = obj.outNames(outputIdx);
            
            % NOTE: Ensure plotParityFig and plotResidualDiagnosticsFig 
            % are also updated to support LaTeX formatting if possible.
            plotParityFig(yTrue, yPred, m, name, obj.fontSize, ...
                obj.exportPath, obj.exportFormat);
                
            plotResidualDiagnosticsFig(yTrue, yPred, m, name, obj.fontSize, ...
                obj.exportPath, obj.exportFormat);
        end
        
        function plot1DSlices(obj, outputIdx, options)
            % plot1DSlices  Plots parametric 1D sweeps holding other variables at OpPoint.
            arguments
                obj
                outputIdx         (1,1) double = 1
                options.numSlices (1,1) double = 50
            end

            outName = obj.outNames(outputIdx);
            nDim    = numel(obj.inNames);
            nFid    = numel(obj.trainSets);

            fig = figure("Name", sprintf("1D Slices — %s", outName), "NumberTitle", "off", "Color", "w");
            nCols = min(nDim, 3);
            nRows = ceil(nDim / nCols);
            tl = tiledlayout(fig, nRows, nCols, "TileSpacing", "compact", "Padding", "compact");

            % Build fixed parameters string for subtitle
            fixedStrs = arrayfun(@(i) sprintf('%s=%.3g', obj.formatLabel(obj.inNames(i), false, true), obj.opReal(i)), ...
                1:nDim, 'UniformOutput', false);

            for iFeat = 1:nDim
                ax = nexttile(tl);
                hold(ax, "on"); grid(ax, "on"); ax.GridAlpha = 0.3; ax.Box = "off";

                % Sweeps
                uSweep = linspace(0, 1, options.numSlices)';
                rSweep = obj.mapToReal(uSweep, iFeat);

                % Build query matrix: base is OpPoint, replace iFeat with sweep
                xQ = repmat(obj.opUnit, options.numSlices, 1);
                xQ(:, iFeat) = uSweep;

                % Predict Mean and Variance
                [yFull, varFull] = obj.predictFcn(xQ);

                yMean = yFull(:, outputIdx);

                % Plot Uncertainty Ribbon
                yStd = sqrt(max(0, varFull(:, outputIdx))); % Max for numerical safety
                yUpper = yMean + 3 * yStd;
                yLower = yMean - 3 * yStd;

                xFill = [rSweep; flipud(rSweep)];
                yFill = [yUpper; flipud(yLower)];

                % Transparent fill for 3-sigma confidence bounds
                fill(ax, xFill, yFill, [0.5 0.5 0.5], ...
                    "FaceAlpha", 0.3, "EdgeColor", "none", ...
                    "DisplayName", "$\pm 3\sigma$ Confidence");

                % Plot Mean Curve
                plot(ax, rSweep, yMean, "-", "LineWidth", 2.0, "Color", "k", "DisplayName", "Surrogate");

                % Overlay Training Data
                for iFid = 1:nFid
                    ds = obj.trainSets(iFid);
                    idx = obj.getSubsampleIdx(size(ds.xReal, 1));
                    xPlot = ds.xReal(idx, iFeat);
                    yPlot = ds.y(idx, outputIdx);
                    c  = obj.getFidelityColor(iFid, nFid);
                    m  = obj.getFidelityMarker(iFid, nFid);
                    name = sprintf("Train Fid %d", iFid);
                    if nFid == 2 && iFid == 1, name = "Train LF"; end
                    if nFid == 2 && iFid == 2, name = "Train HF"; end

                    % Scatter train points
                    scatter(ax, xPlot, yPlot, 30, ...
                        "Marker", m, "MarkerFaceColor", c, "MarkerEdgeColor", "none", ...
                        "MarkerFaceAlpha", 0.6, "DisplayName", name);

                    % Vertical "error" lines (Train Y vs Model Prediction at Train X)
                    [yPredTrainFull, ~] = obj.predictFcn(ds.xUnit(idx, :));

                    yPredTrain = yPredTrainFull(:, outputIdx);
                    lineStyle = ":"; if iFid == nFid, lineStyle = "-"; end

                    plot(ax, [xPlot, xPlot]', [yPlot, yPredTrain]', ...
                        lineStyle, "Color", [c, 0.4], "LineWidth", 1.0, "HandleVisibility", "off");
                end

                % Labels
                xlabel(ax, obj.formatLabel(obj.inNames(iFeat), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                ylabel(ax, obj.formatLabel(outName, false), "Interpreter", "latex", "FontSize", obj.fontSize);
                set(ax, 'TickLabelInterpreter', 'latex');

                % Subtitle indicating fixed variables
                fixedIdx = setdiff(1:nDim, iFeat);
                if ~isempty(fixedIdx)
                    subStr = strjoin(fixedStrs(fixedIdx), ', ');
                    title(ax, sprintf("Fixed: $%s$", subStr), "Interpreter", "latex", "FontSize", obj.fontSize - 2);
                end

                if iFeat == 1
                    lgd = legend(ax, "Location", "best", "Interpreter", "latex");
                    lgd.Box = "off";
                end
            end

            title(tl, sprintf("1D Parametric Slices: \\textbf{%s}", obj.formatLabel(outName, false)), ...
                "Interpreter", "latex", "FontSize", obj.fontSize + 2);

            if strlength(obj.exportPath) > 0
                exportFigure(fig, "1DSlices_" + outName, obj.exportPath, obj.exportFormat);
            end
        end

        function plot2DContours(obj, outputIdx, options)
            % plot2DContours  Generates 2D contour maps for predictive mean and uncertainty.
            %
            % Creates a multi-row tiled layout. For each valid pair of inputs,
            % plots the Mean response on the left and the Predictive Std on the right.
            arguments
                obj
                outputIdx         (1,1) double = 1
                options.res       (1,1) double = 100 % Higher resolution for smooth contours
                options.maxPairs  (1,1) double = 6
            end

            outName = obj.outNames(outputIdx);
            nDim    = numel(obj.inNames);
            nFid    = numel(obj.trainSets);

            pairs  = nchoosek(1:nDim, 2);
            nPairs = min(size(pairs, 1), options.maxPairs);
            if nPairs == 0, return; end

            fig = figure("Name", sprintf("2D Contours — %s", outName), "NumberTitle", "off", "Color", "w");
            % 2 columns: Left = Mean, Right = Uncertainty
            tl = tiledlayout(fig, nPairs, 2, "TileSpacing", "compact", "Padding", "compact");

            % Generate base operating point matrices
            uSweep = linspace(0, 1, options.res);
            [U1, U2] = meshgrid(uSweep, uSweep);
            nQ = numel(U1);

            for p = 1:nPairs
                i1 = pairs(p, 1);
                i2 = pairs(p, 2);

                % Build query
                xQ = repmat(obj.opUnit, nQ, 1);
                xQ(:, i1) = U1(:);
                xQ(:, i2) = U2(:);

                % Predict Mean and Variance safely
                try
                    [yFull, varFull] = obj.predictFcn(xQ);
                catch
                    yFull = obj.predictFcn(xQ);
                    varFull = zeros(size(yFull)); % Fallback if variance fails
                end

                ZMean = reshape(yFull(:, outputIdx), options.res, options.res);
                ZStd  = reshape(sqrt(max(0, varFull(:, outputIdx))), options.res, options.res);

                % Real Coordinates for plotting
                R1 = obj.mapToReal(U1, i1);
                R2 = obj.mapToReal(U2, i2);

                % ---------------------------------------------------------
                % LEFT TILE: Predictive Mean
                % ---------------------------------------------------------
                axMean = nexttile(tl);
                hold(axMean, "on"); axMean.Box = "on";

                contourf(axMean, R1, R2, ZMean, 40, "LineColor", "none");
                colormap(axMean, turbo); % Great for aerodynamics
                colorbar(axMean, "TickLabelInterpreter", "latex");

                % Overlay training points (2D scatter)
                for iFid = 1:nFid
                    ds = obj.trainSets(iFid);
                    idx = obj.getSubsampleIdx(size(ds.xReal, 1));
                    x1C = ds.xReal(idx, i1);
                    x2C = ds.xReal(idx, i2);
                    c = obj.getFidelityColor(iFid, nFid);
                    m = obj.getFidelityMarker(iFid, nFid);
                    name = sprintf("Train Fid %d", iFid);
                    if nFid == 2 && iFid == 1, name = "Train LF"; end
                    if nFid == 2 && iFid == 2, name = "Train HF"; end

                    scatter(axMean, x1C, x2C, 35, m, "filled", ...
                        "MarkerFaceColor", c, "MarkerEdgeColor", "k", ...
                        "DisplayName", name);
                end

                xlabel(axMean, obj.formatLabel(obj.inNames(i1), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                ylabel(axMean, obj.formatLabel(obj.inNames(i2), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                set(axMean, 'TickLabelInterpreter', 'latex');
                if p == 1
                    title(axMean, "Predictive Mean $\mu(\mathbf{x})$", "Interpreter", "latex", "FontSize", obj.fontSize + 1);
                    lgd = legend(axMean, "Location", "best", "Interpreter", "latex"); lgd.Box = "off";
                end

                % ---------------------------------------------------------
                % RIGHT TILE: Predictive Standard Deviation
                % ---------------------------------------------------------
                axStd = nexttile(tl);
                hold(axStd, "on"); axStd.Box = "on";

                contourf(axStd, R1, R2, ZStd, 40, "LineColor", "none");
                colormap(axStd, hot); % Highlights the regions of high uncertainty
                colorbar(axStd, "TickLabelInterpreter", "latex");

                % Overlay training points (to show variance goes to 0 near them)
                for iFid = 1:nFid
                    ds = obj.trainSets(iFid);
                    idx = obj.getSubsampleIdx(size(ds.xReal, 1));
                    x1C = ds.xReal(idx, i1);
                    x2C = ds.xReal(idx, i2);
                    c = obj.getFidelityColor(iFid, nFid);
                    m = obj.getFidelityMarker(iFid, nFid);
                    scatter(axStd, x1C, x2C, 35, m, "filled", ...
                        "MarkerFaceColor", c, "MarkerEdgeColor", "k", "HandleVisibility", "off");
                end

                xlabel(axStd, obj.formatLabel(obj.inNames(i1), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                ylabel(axStd, obj.formatLabel(obj.inNames(i2), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                set(axStd, 'TickLabelInterpreter', 'latex');
                if p == 1
                    title(axStd, "Predictive Std. Dev. $\sigma(\mathbf{x})$", "Interpreter", "latex", "FontSize", obj.fontSize + 1);
                end
            end

            title(tl, sprintf("2D Contour Slices: \\textbf{%s}", obj.formatLabel(outName, false)), ...
                "Interpreter", "latex", "FontSize", obj.fontSize + 2);

            if strlength(obj.exportPath) > 0
                exportFigure(fig, "2DContours_" + outName, obj.exportPath, obj.exportFormat);
            end
        end

        function plot3DSurfaces(obj, outputIdx, options)
            % plot3DSurfaces  3D physical surfaces with data point projections.
            arguments
                obj
                outputIdx         (1,1) double = 1
                options.res       (1,1) double = 40
                options.maxPairs  (1,1) double = 6
            end
            
            outName = obj.outNames(outputIdx);
            nDim    = numel(obj.inNames);
            nFid    = numel(obj.trainSets);
            
            pairs  = nchoosek(1:nDim, 2);
            nPairs = min(size(pairs, 1), options.maxPairs);
            if nPairs == 0, return; end
            
            fig = figure("Name", sprintf("3D Surfaces — %s", outName), "NumberTitle", "off", "Color", "w");
            nCols = min(nPairs, 3);
            nRows = ceil(nPairs / nCols);
            tl = tiledlayout(fig, nRows, nCols, "TileSpacing", "compact", "Padding", "compact");
            
            % Generate base operating point matrices
            uSweep = linspace(0, 1, options.res);
            [U1, U2] = meshgrid(uSweep, uSweep);
            nQ = numel(U1);
            
            for p = 1:nPairs
                i1 = pairs(p, 1);
                i2 = pairs(p, 2);
                
                ax = nexttile(tl);
                hold(ax, "on"); grid(ax, "on"); view(ax, -35, 30); ax.Box = "off";
                
                % Build query
                xQ = repmat(obj.opUnit, nQ, 1);
                xQ(:, i1) = U1(:);
                xQ(:, i2) = U2(:);
                
                % Predict
                yFull = obj.predictFcn(xQ);
                Z     = reshape(yFull(:, outputIdx), options.res, options.res);
                
                % Real Coordinates for plotting
                R1 = obj.mapToReal(U1, i1);
                R2 = obj.mapToReal(U2, i2);
                
                % Surface
                surf(ax, R1, R2, Z, "FaceAlpha", 0.7, "EdgeColor", "none");
                colormap(ax, parula);

                % Data Projections
                for iFid = 1:nFid
                    ds = obj.trainSets(iFid);

                    idx = obj.getSubsampleIdx(size(ds.xReal, 1));
                    x1C = ds.xReal(idx, i1);
                    x2C = ds.xReal(idx, i2);
                    yTrueTrain = ds.y(idx, outputIdx);

                    c  = obj.getFidelityColor(iFid, nFid);
                    m  = obj.getFidelityMarker(iFid, nFid);

                    yFullTrain = obj.predictFcn(ds.xUnit(idx, :));
                    yPredTrain = yFullTrain(:, outputIdx);

                    % Sticks
                    lineStyle = ":"; if iFid == nFid, lineStyle = "-"; end
                    plot3(ax, [x1C, x1C]', [x2C, x2C]', [yTrueTrain, yPredTrain]', ...
                        lineStyle, "Color", [c, 0.4], "LineWidth", 1.0);

                    % Points
                    scatter3(ax, x1C, x2C, yTrueTrain, 25, m, "filled", ...
                        "MarkerFaceColor", c, "MarkerEdgeColor", "k");
                end
                
                % Formatting
                xlabel(ax, obj.formatLabel(obj.inNames(i1), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                ylabel(ax, obj.formatLabel(obj.inNames(i2), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                zlabel(ax, obj.formatLabel(outName, false), "Interpreter", "latex", "FontSize", obj.fontSize);
                set(ax, 'TickLabelInterpreter', 'latex');
            end
            
            cb = colorbar(nexttile(tl, 1)); 
            cb.Layout.Tile = 'east'; cb.TickLabelInterpreter = 'latex';
            title(tl, sprintf("3D Surrogate Surfaces: \\textbf{%s}", obj.formatLabel(outName, false)), ...
                "Interpreter", "latex", "FontSize", obj.fontSize + 2);
                
            if strlength(obj.exportPath) > 0
                exportFigure(fig, "3DSurfaces_" + outName, obj.exportPath, obj.exportFormat);
            end
        end
        
    end
    
    methods (Access = private)
        
        function setupOperatingPoint(obj, meta)
            % Se l'utente passa defaultValues in meta, usali, altrimenti usa 0.5 unitario
            nDim = numel(obj.inNames);
            if isfield(meta, 'defaultValuesUnit') && ~isempty(meta.defaultValuesUnit)
                obj.opUnit = meta.defaultValuesUnit;
                obj.opReal = meta.defaultValuesReal;
            else
                obj.opUnit = repmat(0.5, 1, nDim);
                obj.opReal = arrayfun(@(i) obj.mapToReal(0.5, i), 1:nDim);
            end
        end
        
        function xReal = mapToReal(obj, xUnit, dimIdx)
            % Converts unitary coordinates back to physical bounds
            p = obj.power(dimIdx);
            minR = obj.ranges(1, dimIdx);
            maxR = obj.ranges(2, dimIdx);
            xReal = minR + (xUnit.^(1/p)) .* (maxR - minR);
        end
        
        function c = getFidelityColor(obj, iFid, nFid)
            if nFid == 1, c = obj.cHfTrain; return; end
            if iFid == nFid, c = obj.cHfTrain; else, c = obj.cLfTrain; end
        end
        
        function m = getFidelityMarker(~, iFid, nFid)
            if nFid == 1, m = "hexagram"; return; end
            if iFid == nFid, m = "hexagram"; else, m = "square"; end
        end
        
        function [yMean, yStd] = safeVariance(~, model, xQ)
            try
                [yMean, yStd] = model.predictWithStd(xQ);
            catch
                yMean = model.predict(xQ);
                yStd  = NaN(size(yMean));
            end
        end

        function str = formatLabel(obj, varName, isUnitary, noUnitsStr)
            arguments
                obj
                varName     (1,1) string
                isUnitary   (1,1) logical = false
                noUnitsStr  (1,1) logical = false
            end

            texName = varName;

            if texName == "alpha", texName = '\alpha'; end
            if texName == "beta",  texName = '\beta'; end
            if texName == "phi",   texName = '\phi'; end
            if texName == "mach",  texName = '\mathrm{Mach}'; end
            if texName == "alt",   texName = '\mathrm{Alt}'; end
            if texName == "hprot", texName = 'h_{prot}'; end

            if startsWith(texName, "C") && strlength(texName) > 1
                texName = sprintf('C_{%s}', extractAfter(texName, 1));
            end

            if noUnitsStr
                str = texName;
                return;
            end

            if isUnitary
                str = sprintf('$\\tilde{%s}$ [-]', texName);
            else
                if isfield(obj.units, varName)
                    u = string(obj.units.(varName));
                else
                    u = "-";
                end

                if u == "%"
                    u = '\%';
                end

                str = sprintf('$%s$ [%s]', texName, u);
            end
        end

        function idx = getSubsampleIdx(obj, nPts)
            if nPts <= obj.maxDisplayPts
                idx = 1:nPts;
            else
                idx = round(linspace(1, nPts, obj.maxDisplayPts));
            end
        end
    end
end