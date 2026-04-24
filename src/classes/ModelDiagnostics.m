classdef ModelDiagnostics < handle
% ModelDiagnostics  Orchestrates the diagnostic suite for a CoefficientModel.
%
% This class decouples visualization and performance profiling from the core 
% surrogate model logic. It orchestrates the generation of response surfaces, 
% parity plots, and residual distributions.
%
% DEPENDENCIES:
%   - SurrogatePlotter: Handles per-coefficient aerodynamic plots.
%   - DatasetPlotter: Handles input space and sampling distributions.
%   - calculateMetrics: External function to compute RMSE, R2, MAPE, MaxAE.
%
% USAGE:
%   diag = ModelDiagnostics(coeffModel, "savePlots", true, "exportPath", "results/run01");
%   diag.runDiagnostics();
%
%   diag.plotConvergence(@(n) buildModel(n), [200 400 800]);

    properties (Access = private)
        coeffModel                   % Handle to the main CoefficientModel
        primary                      % Primary dataset (HF or single fidelity)
        trainSets    (1,:) struct    % Array of training sets (LF, HF)
        testSets     (1,:) struct    % Array of testing sets (LF, HF)
        
        % Configuration Options
        savePlots    (1,1) logical = false
        exportPath   (1,1) string  = ""
        exportFormat (1,1) string  = "png"
        plotSurface  (1,1) logical = true
        fontSize     (1,1) double  = 11
        relErrFloor  (1,1) double  = 0.02
    end
    
    methods (Access = public)
        
        function obj = ModelDiagnostics(coeffModel, options)
            % ModelDiagnostics  Constructor.
            arguments
                coeffModel
                options.savePlots    (1,1) logical = false
                options.exportPath   (1,1) string  = ""
                options.exportFormat (1,1) string  = "png"
                options.plotSurface  (1,1) logical = true
                options.fontSize     (1,1) double  = 11
                options.relErrFloor  (1,1) double  = 0.02
            end
            
            obj.coeffModel   = coeffModel;
            obj.savePlots    = options.savePlots;
            obj.exportPath   = options.exportPath;
            obj.exportFormat = options.exportFormat;
            obj.plotSurface  = options.plotSurface;
            obj.fontSize     = options.fontSize;
            obj.relErrFloor  = options.relErrFloor;
            
            obj.primary = coeffModel.getPrimaryDataset();
            [obj.trainSets, obj.testSets] = obj.buildSplits();
        end
        
        function runDiagnostics(obj)
            % runDiagnostics  Executes the full textual and visual pipeline.
            % Prints timings and metrics to the console, then generates dataset
            % and surrogate plots.
            obj.printTimings();
            obj.printMetricsSummary();
            obj.plotDatasets();
            obj.plotSurrogates();
        end
        
        function plotDatasets(obj)
            % plotDatasets  Delegates sampling and distribution plots to DatasetPlotter.
            % Loops through all available fidelities (LF, HF, or single).
            datasets = obj.getAvailableDatasets();
            
            % Determine base export path if saving is enabled
            basePath = "";
            if obj.savePlots
                basePath = obj.exportPath;
            end
            
            for k = 1:numel(datasets)
                tag = sprintf("Dataset_%s", datasets(k).tag);
                currPath = "";
                if obj.savePlots
                    currPath = fullfile(basePath, tag);
                end
                
                p = DatasetPlotter(datasets(k).ds, ...
                    "datasetTag",   datasets(k).tag, ... 
                    "units",        obj.coeffModel.datasetUnits, ... 
                    "exportPath",   currPath, ...
                    "exportFormat", obj.exportFormat, ...
                    "fontSize",     obj.fontSize);
                p.plotAll();
            end
        end

        function plotSurrogates(obj)
            % plotSurrogates  Delegates regression and surface plots to SurrogatePlotter.
            % Generates visuals for every aerodynamic coefficient predicted by the model.
            meta = struct( ...
                'inNames',  {obj.coeffModel.inNames},  ...
                'outNames', {obj.coeffModel.outNames}, ...
                'ranges',   obj.primary.ranges,        ...
                'power',    obj.primary.power);

            testAgg = obj.testSets(end);

            ePath = "";
            if obj.savePlots
                ePath = obj.exportPath;
            end

            p = SurrogatePlotter(obj.coeffModel.model, obj.trainSets, testAgg, meta, ...
                "units",        obj.coeffModel.datasetUnits, ...
                "exportPath",   ePath, ...
                "exportFormat", obj.exportFormat, ...
                "fontSize",     obj.fontSize);

            fprintf('\n=== Surrogate Plot Generation ===\n');
            for j = 1:numel(obj.coeffModel.outNames)
                coeffName = obj.coeffModel.outNames(j);
                fprintf("  -> Rendering plots for %s...\n", coeffName);

                p.plotResiduals(j);

                if obj.plotSurface
                    p.plot1DSlices(j);
                    p.plot2DContours(j);
                    p.plot3DSurfaces(j);
                end
            end
            fprintf('=================================\n');
        end

        function printMetricsSummary(obj)
            % printMetricsSummary  Prints clean ASCII tables of regression and variance metrics.
            testSet = obj.testSets(end);
            nOut    = numel(obj.coeffModel.outNames);

            % --- 1. Standard Regression Metrics Table ---
            fprintf('\n=== Standard Regression Metrics (Test Set) ===\n');
            fprintf('%-10s | %-8s | %-8s | %-8s | %-8s\n', 'Coeff', 'RMSE', 'R^2', 'MAPE [%]', 'MaxAE');
            fprintf('%s\n', repmat('-', 1, 55));
            for j = 1:nOut
                yTrue = testSet.y(:, j);

                % Fail-safe prediction call
                [yFull, varFull] = obj.coeffModel.model.predict(testSet.xUnit);

                yPred = yFull(:, j);
                m = calculateMetrics(yTrue, yPred, obj.relErrFloor);

                fprintf('%-10s | %8.4f | %8.4f | %8.2f | %8.4f\n', ...
                    obj.coeffModel.outNames(j), m.rmse, m.r2, m.mape, m.maxAE);
            end
            fprintf('%s\n', repmat('-', 1, 55));

            % --- 2. Predictive Variance Diagnostics Table ---
            % Check if variance prediction is available and successfully evaluated
            if ~isempty(varFull)
                fprintf('\n=== Predictive Variance Diagnostics ===\n');
                fprintf('%-10s | %-8s | %-8s | %-8s\n', 'Coeff', 'MSSE', 'Cov 95%', 'Cov 99.7%');
                fprintf('%s\n', repmat('-', 1, 45));
                for j = 1:nOut
                    yTrue = testSet.y(:, j);
                    yPred = yFull(:, j);

                    % Safety max(0, var) to avoid complex numbers from numerical noise
                    vPred = max(0, varFull(:, j));
                    stdPred = sqrt(vPred);
                    err = yTrue - yPred;

                    % 1. Mean Standardized Squared Error (Ideal = 1.0)
                    % Added eps to denominator to prevent division by zero
                    msse = mean((err.^2) ./ (vPred + eps));

                    % 2. Coverage Probabilities (Ideal = 95% and 99.7%)
                    cov95 = sum(abs(err) <= 1.96 * stdPred) / length(yTrue) * 100;
                    cov99 = sum(abs(err) <= 3.00 * stdPred) / length(yTrue) * 100;

                    fprintf('%-10s | %8.4f | %7.2f%% | %7.2f%%\n', ...
                        obj.coeffModel.outNames(j), msse, cov95, cov99);
                end
                fprintf('%s\n', repmat('-', 1, 45));
            end
        end
        
        function printTimings(obj)
            % printTimings  Benchmarks inference speed and logs training time.
            trainTime = obj.coeffModel.model.trainingTime;
            
            % Benchmark inference using 10,000 queries
            nBench = 1e4;
            xBench = rand(nBench, numel(obj.coeffModel.inNames));
            
            tInf = tic;
            obj.coeffModel.model.predict(xBench);
            inferenceTimeTotal = toc(tInf);
            inferenceTimePerPt = inferenceTimeTotal / nBench;
            
            fprintf('\n=== Timing Diagnostics ===\n');
            fprintf('  Training time      : %7.3f s\n', trainTime);
            fprintf('  Inference (10k)    : %7.3f s\n', inferenceTimeTotal);
            fprintf('  Inference/point    : %7.2f μs\n', inferenceTimePerPt * 1e6);
            fprintf('%s\n', repmat('=', 1, 26));
        end
        
        function plotConvergence(obj, builderFcn, sweepVec, sweepLabel)
            % plotConvergence  Evaluates and plots metric convergence over a parameter sweep.
            %
            % INPUTS:
            %   builderFcn : Function handle returning a trained model given a sweep parameter
            %   sweepVec   : Array of parameter values to test (e.g., sample sizes)
            %   sweepLabel : String labeling the x-axis of the convergence plot
            arguments
                obj
                builderFcn  (1,1) function_handle
                sweepVec    (1,:) double
                sweepLabel  (1,1) string = "N training points"
            end
            
            ePath = "";
            if obj.savePlots
                ePath = obj.exportPath; 
            end
            
            plotMetricsConvergence(builderFcn, sweepVec, ...
                "sweepLabel",   sweepLabel,       ...
                "exportPath",   ePath,   ...
                "exportFormat", obj.exportFormat, ...
                "fontSize",     obj.fontSize,     ...
                "relErrFloor",  obj.relErrFloor);
        end
        
    end
    
    methods (Access = private)
        
        function [trainSets, testSets] = buildSplits(obj)
            % buildSplits  Extracts training and testing splits per fidelity.
            cm = obj.coeffModel;
            if cm.datasetMode == "multiFidelity"
                [~, trainLF, testLF] = cm.datcomDataset.split();
                [~, trainHF, testHF] = cm.cfdDataset.split();
                trainSets = [trainLF, trainHF];
                testSets  = [testLF, testHF];  
            else
                [~, trainSets, testSets] = obj.primary.split();
            end
        end
        
        function datasets = getAvailableDatasets(obj)
            % getAvailableDatasets  Aggregates active datasets into a struct array.
            cm       = obj.coeffModel;
            datasets = struct('tag', {}, 'ds', {});
            
            if ~isempty(cm.datcomDataset) && cm.datcomDataset.isReady
                datasets(end+1) = struct('tag', "LF_datcom", 'ds', cm.datcomDataset);
            end
            
            if ~isempty(cm.cfdDataset) && cm.cfdDataset.isReady
                datasets(end+1) = struct('tag', "HF_cfd", 'ds', cm.cfdDataset);
            end
            
            if isempty(datasets)
                datasets(1) = struct('tag', "primary", 'ds', obj.primary);
            end
        end
        
    end
end