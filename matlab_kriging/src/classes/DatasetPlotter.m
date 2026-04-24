classdef DatasetPlotter < handle
% DatasetPlotter  Visualization of dataset sample distribution and normalization.
%
% Produces minimalist, publication-ready plots using LaTeX formatting.
% Extracts units dynamically to differentiate between physical and unitary domains.
%
% USAGE:
%   p = DatasetPlotter(dataset, "datasetTag", "HF_cfd", "units", datasetUnits);
%   p.plotAll();

    properties (Access = private)
        % Data
        xUnit    (:,:) double
        xReal    (:,:) double
        y        (:,:) double
        trainIdx (:,1) logical
        testIdx  (:,1) logical
        ranges   (2,:) double
        power    (1,:) double
        inNames  (1,:) string
        outNames (1,:) string
        
        % Metadata & Styling
        datasetTag   (1,1) string
        units        (1,1) struct
        exportPath   (1,1) string  = ""
        exportFormat (1,1) string  = "png"
        maxPairs     (1,1) double  = 6
        fontSize     (1,1) double  = 11
        
        % Colors (Minimalist Palette)
        cTrain = [0.25 0.41 0.67] % Soft Navy
        cTest  = [0.85 0.33 0.10] % Deep Orange
        cKde   = [0.15 0.25 0.45] % Darker Navy for KDE lines
    end
    
    methods
        function obj = DatasetPlotter(dataset, options)
            arguments
                dataset
                options.datasetTag   (1,1) string  = "Primary"
                options.units        (1,1) struct  = struct()
                options.exportPath   (1,1) string  = ""
                options.exportFormat (1,1) string  = "png"
                options.maxPairs     (1,1) double  = 6
                options.fontSize     (1,1) double  = 11
            end
            
            obj.xUnit    = dataset.xUnit;
            obj.xReal    = dataset.xReal;
            obj.y        = dataset.y;
            obj.trainIdx = dataset.trainIdx;
            obj.testIdx  = dataset.testIdx;
            obj.ranges   = dataset.ranges;
            obj.power    = dataset.power;
            
            obj.inNames  = string(dataset.metadata.inNames);
            obj.outNames = string(dataset.metadata.outNames);
            
            obj.datasetTag   = strrep(options.datasetTag, "_", "\_"); % Escape per LaTeX
            obj.units        = options.units;
            obj.exportPath   = options.exportPath;
            obj.exportFormat = options.exportFormat;
            obj.maxPairs     = options.maxPairs;
            obj.fontSize     = options.fontSize;
        end
        
        function plotAll(obj)
            obj.plotSampleSpace();
            obj.plotOutputDistribution();
            obj.plotNormalization();
        end
        
        function plotSampleSpace(obj)
            % Scatter plots in physical space. Features a global shared legend
            % and colorbar attached to the tiledlayout, instead of a single axes.
            nDim   = numel(obj.inNames);
            pairs  = nchoosek(1:nDim, 2);
            nPairs = min(size(pairs, 1), obj.maxPairs);
            if nPairs == 0, return; end
            
            nCols = min(nPairs, 3);
            nRows = ceil(nPairs / nCols);
            
            fig = figure("Name", sprintf("Sample Space [%s]", obj.datasetTag), ...
                         "NumberTitle", "off", "Color", "w");
            tl  = tiledlayout(fig, nRows, nCols, "TileSpacing", "compact", "Padding", "compact");
            
            hasSplit = ~isempty(obj.trainIdx) && ~isempty(obj.testIdx);
            cVal     = obj.y(:, 1); % Color by primary output
            
            % Generate all subplots
            for p = 1:nPairs
                i1 = pairs(p, 1);
                i2 = pairs(p, 2);
                x1 = obj.xReal(:, i1);
                x2 = obj.xReal(:, i2);
                
                ax = nexttile(tl, p);
                hold(ax, "on"); 
                grid(ax, "on");
                ax.GridAlpha = 0.3;
                ax.Box = "off"; % Minimalist look
                
                if hasSplit
                    scatter(ax, x1(obj.trainIdx), x2(obj.trainIdx), 25, cVal(obj.trainIdx), ...
                        "filled", "MarkerFaceAlpha", 0.7, "MarkerEdgeColor", "none", ...
                        "DisplayName", "Train");
                    scatter(ax, x1(obj.testIdx), x2(obj.testIdx), 35, cVal(obj.testIdx), ...
                        "hexagram", "MarkerEdgeColor", obj.cTest, "LineWidth", 1.2, ...
                        "DisplayName", "Test");
                else
                    scatter(ax, x1, x2, 25, cVal, "filled", ...
                        "MarkerFaceAlpha", 0.8, "MarkerEdgeColor", "none");
                end
                
                % LaTeX Labels with Units
                xlabel(ax, obj.formatLabel(obj.inNames(i1), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                ylabel(ax, obj.formatLabel(obj.inNames(i2), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', obj.fontSize - 1);
                
                xlim(ax, obj.ranges(:, i1)');
                ylim(ax, obj.ranges(:, i2)');
            end
            
            colormap(fig, turbo);
            
            % --- GLOBAL DECORATIONS ---
            ax1 = nexttile(tl, 1);
            
            % Global Colorbar on the East
            cb = colorbar(ax1);
            cb.Layout.Tile = 'east';
            cb.TickLabelInterpreter = 'latex';
            cb.Label.Interpreter = 'latex';
            cb.Label.String = obj.formatLabel(obj.outNames(1), false);
            cb.Label.FontSize = obj.fontSize;
            
            % Global Legend on the South
            if hasSplit
                lgd = legend(ax1, "Orientation", "horizontal", "Interpreter", "latex");
                lgd.Layout.Tile = 'south';
                lgd.Box = "off";
                lgd.FontSize = obj.fontSize;
            end
            
            title(tl, sprintf("Sample Distribution: \\textbf{%s}", obj.datasetTag), ...
                "Interpreter", "latex", "FontSize", obj.fontSize + 2);
            
            if strlength(obj.exportPath) > 0
                exportFigure(fig, "SampleSpace", obj.exportPath, obj.exportFormat);
            end
        end
        
        function plotOutputDistribution(obj)
            % Histogram + Kernel Density Estimate (KDE) for a sleek, modern look.
            nOut  = numel(obj.outNames);
            nCols = min(nOut, 3);
            nRows = ceil(nOut / nCols);
            
            fig = figure("Name", sprintf("Output Dist [%s]", obj.datasetTag), ...
                         "NumberTitle", "off", "Color", "w");
            tl  = tiledlayout(fig, nRows, nCols, "TileSpacing", "compact", "Padding", "compact");
            
            hasSplit = ~isempty(obj.trainIdx);
            
            for j = 1:nOut
                ax = nexttile(tl, j);
                hold(ax, "on"); grid(ax, "on"); ax.GridAlpha = 0.3; ax.Box = "off";
                
                dataAll = obj.y(:, j);
                
                if hasSplit
                    dataTrain = obj.y(obj.trainIdx, j);
                    
                    % Train Histogram (PDF normalized)
                    histogram(ax, dataTrain, 'Normalization', 'pdf', ...
                        'FaceColor', obj.cTrain, 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.5, 'DisplayName', 'Train Dist');
                    
                    % Smooth KDE line for Train
                    [f, xi] = ksdensity(dataTrain);
                    plot(ax, xi, f, 'Color', obj.cKde, 'LineWidth', 1.5, 'DisplayName', 'Train KDE');
                    
                    % Test points as vertical rugs or simple secondary KDE
                    if any(obj.testIdx) 
                        dataTest = obj.y(obj.testIdx, j);
                        if numel(dataTest) > 1 
                            [fTest, xiTest] = ksdensity(dataTest);
                            plot(ax, xiTest, fTest, '--', 'Color', obj.cTest, ...
                                'LineWidth', 1.5, 'DisplayName', 'Test KDE');
                        end
                    end
                else
                    histogram(ax, dataAll, 'Normalization', 'pdf', ...
                        'FaceColor', obj.cTrain, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
                    [f, xi] = ksdensity(dataAll);
                    plot(ax, xi, f, 'Color', obj.cKde, 'LineWidth', 1.5);
                end
                
                xlabel(ax, obj.formatLabel(obj.outNames(j), false), "Interpreter", "latex", "FontSize", obj.fontSize);
                ylabel(ax, "Density PDF", "Interpreter", "latex", "FontSize", obj.fontSize);
                set(ax, 'TickLabelInterpreter', 'latex');
                
                % Only add legend to first tile, map to layout
                if j == 1 && hasSplit
                    lgd = legend(ax, "Orientation", "horizontal", "Interpreter", "latex");
                    lgd.Layout.Tile = 'south';
                    lgd.Box = "off";
                end
            end
            
            title(tl, sprintf("Output Density Distributions: \\textbf{%s}", obj.datasetTag), ...
                "Interpreter", "latex", "FontSize", obj.fontSize + 2);
                
            if strlength(obj.exportPath) > 0
                exportFigure(fig, "OutputDistribution", obj.exportPath, obj.exportFormat);
            end
        end
        
        function plotNormalization(obj)
            % Minimalist horizontal bar chart or standardized plot for exponents
            nDim = numel(obj.inNames);
            
            fig = figure("Name", sprintf("Normalization [%s]", obj.datasetTag), ...
                         "NumberTitle", "off", "Color", "w");
            tl  = tiledlayout(fig, 1, 1, "TileSpacing", "compact", "Padding", "compact");
            ax  = nexttile(tl, 1);
            hold(ax, "on"); grid(ax, "on"); ax.GridAlpha = 0.3; ax.Box = "off";
            
            % Plot power exponents as stem/lollipop chart (cleaner than bars)
            stem(ax, 1:nDim, obj.power, "filled", ...
                 "Color", obj.cTrain, "MarkerFaceColor", obj.cTrain, ...
                 "LineWidth", 1.5, "MarkerSize", 6);
            
            yline(ax, 1, "--", "Color", obj.cTest, "LineWidth", 1.2, "Label", "Linear (p=1)", ...
                  "Interpreter", "latex", "LabelHorizontalAlignment", "left");
                  
            % Format ticks with LaTeX symbols
            texLabels = arrayfun(@(n) obj.formatLabel(n, true), obj.inNames, 'UniformOutput', false);
            set(ax, "XTick", 1:nDim, "XTickLabel", texLabels, ...
                "TickLabelInterpreter", "latex", "FontSize", obj.fontSize);
            
            ylabel(ax, "Power Exponent $p$", "Interpreter", "latex", "FontSize", obj.fontSize);
            title(tl, sprintf("Input Space Transformation: \\textbf{%s}", obj.datasetTag), ...
                "Interpreter", "latex", "FontSize", obj.fontSize + 2);
            
            if strlength(obj.exportPath) > 0
                exportFigure(fig, "Normalization", obj.exportPath, obj.exportFormat);
            end
        end
    end
    
    methods (Access = private)
        function str = formatLabel(obj, varName, isUnitary)
            texName = varName;

            if texName == "alpha", texName = '\alpha'; end
            if texName == "beta",  texName = '\beta'; end
            if texName == "phi",   texName = '\phi'; end
            if texName == "mach",  texName = '\mathrm{Mach}'; end 
            if texName == "alt",   texName = '\mathrm{Alt}'; end
            if texName == "hprot", texName = 'h_{prot}'; end

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

                if startsWith(texName, "C") && strlength(texName) > 1
                    texName = sprintf('C_{%s}', extractAfter(texName, 1));
                end
                str = sprintf('$%s$ [%s]', texName, u);
            end
        end
    end
end


