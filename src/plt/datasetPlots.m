function [R2, RMSE, MAE] = datasetPlots(predictor, dataset, trainDataset, testDataset, settingsFun)
% datasetPlots
% VISUALIZATION OF KRIGING PREDICTOR VS TARGET FUNCTION
%
% This function generates plots to compare the Kriging surrogate model's
% predictions against the true target function, including a scatter plot
% of true vs. predicted values and surface plots/slices for 2D/3D inputs.
% It also overlays the training and test data points on the surface plots
% for context.
%
% Inputs:
% - dataset: Structure containing input/output data (X, y, grid info, xTrain, xTest).
% - settingsFun: Structure with configuration details (dimensions, grid resolution).
%
% Required variables in workspace (not inputs):
% - yTest: True output values for the test points (assumed available).
% - yPred: Predicted output values for the test points (assumed available).
% - R2: R-squared value of the prediction (assumed available).
% - predictor: A function handle to the Kriging model's predictor function.


xTrain = trainDataset.xUnit;  
xTest  = testDataset.xUnit; 
yTest  = testDataset.y; 

yPred = arrayfun(@(i) predictor(xTest(i,:)), 1: size(xTest, 1))';

residuals = yTest - yPred;
RMSE = sqrt(mean(residuals.^2));
R2 = 1 - sum(residuals.^2) / sum((yTest - mean(yTest)).^2);
MAE = mean(abs(residuals));
fprintf('RMSE: %.4f | R^2: %.4f | MAE: %.4f\n', RMSE, R2, MAE);

% --- 1. Kriging Predictor Performance Scatter Plot (Always Generated) ---
figure;
scatter(yTest, yPred, 'filled'); % Assuming yTest and yPred are fields in dataset
xlabel('True');
ylabel('Predicted');
title(sprintf('Kriging Predictor Performance (R^2=%.3f)', R2));
grid on;
axis equal;
refline(1,0); % Add y=x reference line 

roundedUnitSpan = dataset.grid.roundedUnitSpan;
dimNames = settingsFun.fltcon.dimNames;

% --- 2. Build Grid Vectors in Unit Hypercube ---
nDim = size(dataset.xUnit, 2);
gridVectors = cell(1, nDim);
i = 1;
labels = cell(nDim, 1);

% Identify the dimensions that vary (n ~= 1) and build their grid vectors
for j = 1: numel(dimNames)
    d = dimNames{j};
    if settingsFun.fltcon.(d).n ~= 1
        labels{i} = d;
        gridVectors{i} = roundedUnitSpan(j, 1: settingsFun.fltcon.(d).n);
        i = i + 1;
    end
end

% --- 3. Generate Comparison Plots (2D Case) ---
if nDim == 2
    fprintf('Generating 3D comparison plots...\n');
    xVec = gridVectors{1};
    yVec = gridVectors{2};
    nx = numel(xVec);
    ny = numel(yVec);

    % Generate the 3D grid of test points in the unit hypercube
    [Xq, Yq] = meshgrid(xVec, yVec);
    testGridPoints = [Xq(:), Yq(:)];
    
    % Evaluate the Kriging predictor on all grid points
    yPredGrid = arrayfun(@(i) predictor(testGridPoints(i,:)), 1: size(testGridPoints, 1))';
    
    % Reshape the results for 3D plotting
    predictedCA = reshape(yPredGrid, ny, nx); % Note: Reshape order is (Y, X)
    
    % Target Function (already computed and stored)
    targetCA = dataset.grid.funCA; % Target is [nx, ny]
    targetCA = permute(targetCA, [2 1]); % Transpose to [ny, nx] for alignment

    % Find the min/max values across all surfaces for consistent coloring (optional but good practice)
    minVal = min([targetCA(:); predictedCA(:)]);
    maxVal = max([targetCA(:); predictedCA(:)]);

    % Plot 1: Target Function Surface
    figure;
    surf(xVec, yVec, targetCA, 'EdgeColor', 'none');
    hold on;
    
    % Overlay Training Points (red circles)
    scatter3(xTrain(:, 1), xTrain(:, 2), maxVal * ones(size(xTrain, 1), 1) + 0.1, ...
             50, 'r', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Train Points');
    
    % Overlay Test Points (black diamonds)
    scatter3(xTest(:, 1), xTest(:, 2), maxVal * ones(size(xTest, 1), 1) + 0.1, ...
             50, 'k', 'd', 'filled', 'MarkerEdgeColor', 'w', 'DisplayName', 'Test Points');

    shading interp;
    colorbar;
    clim([minVal maxVal]); % Set color scale
    xlabel(dimNames{1},'Interpreter','none');
    ylabel(dimNames{2},'Interpreter','none');
    zlabel('Output Value');
    axis tight;
    sgtitle('Target Function (InterpCAfun) Surface');
    hold off;

    % Plot 2: Kriging Predictor Surface
    figure;
    surf(xVec, yVec, predictedCA, 'EdgeColor', 'none');
    hold on;
    
    % Overlay Training Points (red circles)
    scatter3(xTrain(:, 1), xTrain(:, 2), maxVal * ones(size(xTrain, 1), 1) + 0.1, ...
             50, 'r', 'filled', 'MarkerEdgeColor', 'k');
    
    % Overlay Test Points (black diamonds)
    scatter3(xTest(:, 1), xTest(:, 2), maxVal * ones(size(xTest, 1), 1) + 0.1, ...
             50, 'k', 'd', 'filled', 'MarkerEdgeColor', 'w');

    shading interp;
    colorbar;
    clim([minVal maxVal]); % Set color scale
    xlabel(dimNames{1},'Interpreter','none');
    ylabel(dimNames{2},'Interpreter','none');
    zlabel('Predicted Value');
    axis tight;
    sgtitle('Kriging Surrogate Model Prediction Surface');
    hold off;
    
    % Plot 3: Difference (Error) Surface
    figure;
    error = targetCA - predictedCA;
    surf(xVec, yVec, error, 'EdgeColor', 'none');
    shading interp;
    view(2); % View from top down for error visualization
    colorbar;
    clim([-max(abs(error(:))), max(abs(error(:)))]); % Symmetric color scale for error
    xlabel(dimNames{1},'Interpreter','none');
    ylabel(dimNames{2},'Interpreter','none');
    zlabel('Error Value');
    axis tight;
    sgtitle('Kriging Prediction Error Surface (Target - Predicted)');

% --- 4. Generate Comparison Plots (3D Case) ---
elseif nDim == 3
    fprintf('Generating 3D comparison plots...\n');
    xVec = gridVectors{1};
    yVec = gridVectors{2};
    zVec = gridVectors{3};
    nx = numel(xVec);
    ny = numel(yVec);
    nz = numel(zVec);

    % Generate the 3D grid of test points in the unit hypercube
    [Xq, Yq, Zq] = meshgrid(xVec, yVec, zVec);
    testGridPoints = [Xq(:), Yq(:), Zq(:)];
    
    % Evaluate the Kriging predictor on all grid points
    yPredGrid = arrayfun(@(i) predictor(testGridPoints(i,:)), 1: size(testGridPoints, 1))';
    
    % Reshape the results for 3D plotting
    predictedCA = reshape(yPredGrid, ny, nx, nz); % Note: Reshape order is (Y, X, Z)
    
    % Target Function (already computed and stored)
    targetCA = dataset.grid.funCA; % Target is [nx, ny, nz]
    targetCA = permute(targetCA, [2 1 3]); % Transpose to [ny, nx, nz] for alignment
    
    % Choose slices (first, middle, last)
    zIdx = [1, round(nz/2), nz];

    % Find the min/max values across all slices/surfaces
    minVal = min([targetCA(:); predictedCA(:)]);
    maxVal = max([targetCA(:); predictedCA(:)]);

    % Plot 1: Target Function Slices
    figure;
    tiledlayout(1,3);
    for i = 1:3
        k = zIdx(i);
        nexttile;
        surf(xVec, yVec, targetCA(:,:,k), 'EdgeColor', 'none');
        hold on;
        
        % Filter and Overlay Training Points for this slice
        % Points must be close to the current Z-slice value (zVec(k))
        zTol = (zVec(2) - zVec(1)) / 2; % Tolerance is half the step size
        trainSliceIdx = abs(xTrain(:, 3) - zVec(k)) < zTol;
        testSliceIdx = abs(xTest(:, 3) - zVec(k)) < zTol;

        % Training Points
        scatter(xTrain(trainSliceIdx, 1), xTrain(trainSliceIdx, 2), ...
                50, 'r', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Train Points');
        
        % Test Points
        scatter(xTest(testSliceIdx, 1), xTest(testSliceIdx, 2), ...
                50, 'k', 'd', 'filled', 'MarkerEdgeColor', 'w', 'DisplayName', 'Test Points');

        shading interp;
        view(2);
        colorbar;
        clim([minVal maxVal]);
        title(sprintf('Target Fun Slice (z=%.3f)', zVec(k)));
        xlabel(dimNames{1},'Interpreter','none');
        ylabel(dimNames{2},'Interpreter','none');
        axis tight;
        hold off;
    end
    sgtitle('Target Function (InterpCAfun) Slices'); 

    % Plot 2: Kriging Predictor Slices
    figure;
    tiledlayout(1,3);
    for i = 1:3
        k = zIdx(i);
        nexttile;
        surf(xVec, yVec, predictedCA(:,:,k), 'EdgeColor', 'none');
        hold on;
        
        % Filter and Overlay Training/Test Points for this slice
        trainSliceIdx = abs(xTrain(:, 3) - zVec(k)) < zTol;
        testSliceIdx = abs(xTest(:, 3) - zVec(k)) < zTol;

        % Training Points
        scatter(xTrain(trainSliceIdx, 1), xTrain(trainSliceIdx, 2), 50, 'r', 'filled', 'MarkerEdgeColor', 'k');
        
        % Test Points
        scatter(xTest(testSliceIdx, 1), xTest(testSliceIdx, 2), 50, 'k', 'd', 'filled', 'MarkerEdgeColor', 'w');

        shading interp;
        view(2);
        colorbar;
        clim([minVal maxVal]);
        title(sprintf('Kriging Prediction Slice (z=%.3f)', zVec(k)));
        xlabel(dimNames{1},'Interpreter','none');
        ylabel(dimNames{2},'Interpreter','none');
        axis tight;
        hold off;
    end
    sgtitle('Kriging Surrogate Model Prediction Slices');

    % Plot 3: Difference (Error) Slices
    figure;
    tiledlayout(1,3);
    for i = 1:3
        k = zIdx(i);
        errorSlice = targetCA(:,:,k) - predictedCA(:,:,k);
        nexttile;
        surf(xVec, yVec, errorSlice, 'EdgeColor', 'none');
        shading interp;
        view(2);
        colorbar;
        clim([-max(abs(errorSlice(:))), max(abs(errorSlice(:)))]); % Symmetric color scale
        title(sprintf('Prediction Error Slice (z=%.3f)', zVec(k)));
        xlabel(dimNames{1},'Interpreter','none');
        ylabel(dimNames{2},'Interpreter','none');
        axis tight;
    end
    sgtitle('Kriging Prediction Error Slices (Target - Predicted)');
end
end

