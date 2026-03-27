classdef SurrogateModel
    % SurrogateModel
    % -------------------------------------------------------------------------
    % High-level manager for training and predicting with surrogate models.
    % 
    % Supports dynamically resolved tuning and prediction functions:
    %   "kriging"   → krigingTuning / krigingPredictor
    %   "coKriging" → coKrigingTuning / coKrigingPredictor
    %
    % Multi-output design:
    %   • data        : Shared across all outputs (x, y). Internally scales y.
    %   • hyperparams : Struct array (1 x nOut) containing optimized parameters 
    %                   (theta, p, lambda) for each output.
    %   • math        : Struct array (1 x nOut) containing derived numerical 
    %                   elements (corrVector, L, mu) for fast evaluation.
    %   • normStats   : Struct array (1 x nOut) with scaling stats (mu, sigma) 
    %                   for automatic internal output normalization.
    %
    % -------------------------------------------------------------------------

    %% ===================== PUBLIC IMMUTABLE PROPERTIES =====================
    properties (SetAccess = private)

        % Logical identifier (e.g., "coefficient", "apogee")
        modelName         (1,1) string

        % Surrogate type identifier ("kriging" or "coKriging")
        surrogateType     (1,1) string

        % Options struct forwarded to the optimizer
        optimizationOptions     struct

        % Function handle to surrogate tuning routine
        modelTuningFcn    function_handle

        % Function handle to prediction routine
        modelPredictorFcn function_handle

        % Readiness flag: TRUE when model is trained and consistent
        isReady           logical = false
    end

    %% ===================== CONSTANT PRIVATE PROPERTIES =====================
    properties (Constant, Access = private)
        projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))))
        configPath  = fullfile(SurrogateModel.projectRoot, 'config')
        dataPath    = fullfile(SurrogateModel.projectRoot, 'data')
    end

    %% ===================== DEPENDENT PUBLIC PROPERTIES =====================
    properties (Dependent, SetAccess = private)
        % Shared training data struct. 
        % Kriging   : x, y
        % CoKriging : xC, yC, xE, yE
        data            struct

        % Output scaling parameters: (normStates.mu, normStates.sigma)
        normStats       struct
        
        % Optimized hyperparameters {1 x nOutputs}
        hyperparams     struct
        
        % Pre-computed matrices/vectors for fast inference {1 x nOutputs}
        math            struct
        
        % Simulink-ready representation (broadcasted parameters)
        modelCHAD       struct
    end

    %% ===================== PRIVATE BACKING STORAGE =====================
    properties (Access = private)
        data_        struct = struct()
        hyperparams_ struct = []
        math_        struct = []
        normStats_   struct = [] 
        modelCHAD_   struct = struct()
    end

    %% ===================== CONSTRUCTOR =====================
    methods
        function obj = SurrogateModel(modelName, surrogateType, ...
                data, configPath, varsIn, options)
            % Loads or trains the surrogate model.
            %
            % INPUTS
            %   modelName       → string identifier ("apogee", "coefficient")
            %   data            → dataset struct containing normalized inputs/outputs
            %   surrogateType   → "kriging" | "coKriging"
            %   optimizationOpt → struct with GA/optimizer settings and savePath
            %
            % OPTIONS
            %   reTrain         → if true, train without checking for
            %                               existing file
            %   autoTrain       → if true, train without interactive prompt
            %   saveModel       → persist model to file after training
            arguments
                modelName          (1,1) string
                surrogateType      (1,1) string
                data                     struct = []
                configPath         (1,1) string = ""
                varsIn                   struct  = struct()
                options.reTrain          logical = false
                options.autoTrain        logical = false
                options.saveModel        logical = true
            end

            % Assignments trigger updateFcn and checkProperties
            obj.modelName     = modelName;
            obj.surrogateType = surrogateType;
            obj.data          = data;

            % Store only if non-empty; do not assign empty default so that
            % missing optimizationOpt is detectable downstream (errors on train)
            if ~isempty(fieldnames(varsIn))
                obj.optimizationOptions = varsIn;
            else
                % Autoresolve configPath if not explicitly provided
                if strlength(configPath) == 0
                    configPath = fullfile(obj.configPath, "models", ...
                        obj.modelName + "OptimizationConfig");
                end

                % Load settings from the resolved path
                optSettings = Settings(configPath);
                obj.optimizationOptions = optSettings.optimizationOptions;
            end


            obj = obj.loadModel(...
                'autoTrain', options.autoTrain, ...
                'saveModel', options.saveModel, ...
                'reTrain'  , options.reTrain);

            % Final readiness validation
            obj.isReady = obj.checkProperties();
        end
    end

    %% ===================== PUBLIC API =====================
    methods (Access = public)
        
        function yPred = predict(obj, xQuery)
            % Evaluate the surrogate model for all outputs.
            %
            % INPUT
            %   xQuery → Points to evaluate [nPoints x nInputs]
            %
            % OUTPUT
            %   yPred  → Predicted values [nPoints x nOutputs]

            if ~obj.isReady
                error("SurrogateModel:NotReady", ...
                    "Model must be trained and dimensionally consistent before prediction.");
            end

            nOut = length(obj.hyperparams_);
            nPoints = size(xQuery, 1);
            yPred = zeros(nPoints, nOut);

            % Create a temporarily normalized dataset for the internal predictors
            normData = obj.data_;
            for i = 1:nOut
                yMu = obj.normStats_(i).mu;
                ySigma = obj.normStats_(i).sigma;
                
                if strcmpi(obj.surrogateType, "coKriging")
                    normData.yC(:, i) = (normData.yC(:, i) - yMu) / ySigma;
                    normData.yE(:, i) = (normData.yE(:, i) - yMu) / ySigma;
                else
                    normData.y(:, i) = (normData.y(:, i) - yMu) / ySigma;
                end
            end

            % Multi-output loop: independent predictions
            for i = 1:nOut
                % Predict using the normalized dataset
                yPredNorm = obj.modelPredictorFcn(xQuery, normData, ...
                    obj.hyperparams_(i), obj.math_(i));
                
                % Un-scale to return physical targets to the user
                yPred(:, i) = yPredNorm * obj.normStats_(i).sigma + obj.normStats_(i).mu;
            end
        end

        % -------------------------------------------------------------------------
        function obj = fit(obj, options)
            % Train surrogate model for all outputs.
            arguments
                obj               SurrogateModel
                options.saveModel logical = true
            end
            if isempty(fieldnames(obj.data_))
                error("SurrogateModel:NoData", "Training data is missing.");
            end
            fprintf("Training %s %s model...\n", obj.modelName, obj.surrogateType);

            isCoKriging = strcmpi(obj.surrogateType, "coKriging");
            if isCoKriging
                nOut = size(obj.data_.yE, 2);
            else
                nOut = size(obj.data_.y, 2);
            end

            for i = 1:nOut
                fprintf("  -> Tuning output %d of %d...\n", i, nOut);

                % Compute normalization statistics
                if isCoKriging
                    yAll = obj.data_.yE(:, i);
                else
                    yAll = obj.data_.y(:, i);
                end

                nStat.mu    = mean(yAll);
                nStat.sigma = std(yAll);
                if nStat.sigma == 0
                    nStat.sigma = 1;
                end

                tuneOpts = obj.optimizationOptions;

                % ==== GRADIENTS HANDLING (GEK) ====
                if isfield(obj.data_, 'dy2D') && isfield(obj.data_, 'gradMap')
                    gMap = obj.data_.gradMap;
                    gMask = (gMap(:, 1) == i);

                    if any(gMask)
                        rawDy = obj.data_.dy2D(:, gMask);
                        dyDirs = gMap(gMask, 2);

                        % d(yNorm)/dx = (1/sigma) * dy/dx
                        tuneOpts.dy = rawDy / nStat.sigma;
                        tuneOpts.dyDirs = dyDirs(:)';

                        fprintf("     [Gradient Enhanced Kriging activated on %d direction]\n", sum(gMask));
                    end
                end
                % ===================================

                % Normalize targets for tuning and call tuning functions
                if isCoKriging
                    yCNorm = (obj.data_.yC(:, i) - nStat.mu) / nStat.sigma;
                    yENorm = (obj.data_.yE(:, i) - nStat.mu) / nStat.sigma;
                    [hOut, mOut] = obj.modelTuningFcn(...
                        obj.data_.xC, yCNorm, ...
                        obj.data_.xE, yENorm, ...
                        tuneOpts);
                else
                    yNorm = (obj.data_.y(:, i) - nStat.mu) / nStat.sigma;
                    [hOut, mOut] = obj.modelTuningFcn(...
                        obj.data_.x, yNorm, tuneOpts);
                end

                % Cache outputs
                if i == 1
                    hyperCache = repmat(hOut, 1, nOut);
                    mathCache  = repmat(mOut, 1, nOut);
                    normCache  = repmat(nStat, 1, nOut);
                else
                    hyperCache(i) = hOut;
                    mathCache(i)  = mOut;
                    normCache(i)  = nStat;
                end
            end

            % Assigning to public properties
            obj.hyperparams = hyperCache;
            obj.math        = mathCache;
            obj.normStats   = normCache;

            fprintf("Training complete.\n");

            if options.saveModel
                obj.saveModel();
            end
        end

        % -------------------------------------------------------------------------
        function saveModel(obj)
            file = fullfile(obj.dataPath, "models",...
                obj.modelName + "Model.mat");
            model.data        = obj.data_;
            model.hyperparams = obj.hyperparams_;
            model.math        = obj.math_;
            model.normStats   = obj.normStats_;
            save(file, "-struct", "model");
            fprintf("Model saved → %s\n", file);
        end
    end

    %% ===================== PRIVATE LIFECYCLE =====================
    methods (Access = private)
        
        function obj = loadModel(obj, options)
            % Load model if available, otherwise optionally train it.
            arguments
                obj               SurrogateModel
                options.autoTrain logical
                options.saveModel logical
                options.reTrain   logical
            end

            varsIn = obj.optimizationOptions;
            if options.reTrain
                obj = obj.fit('saveModel', options.saveModel);
                return;
            end

            file = fullfile(obj.dataPath, "models", ...
                obj.modelName + "Model.mat");
            success = false;

            if ~isempty(file) && exist(file, 'file')
                loadedData = load(file);
                if isfield(loadedData, "hyperparams") && isfield(loadedData, "math") && isfield(loadedData, "data")
                    isConsistent = false;
                    
                    if isempty(fieldnames(obj.data_))
                        isConsistent = true;
                    else
                        if strcmpi(obj.surrogateType, "coKriging")
                            if isfield(loadedData.data, 'xC') && isfield(loadedData.data, 'xE')
                                isConsistent = isequal(obj.data_.xC, loadedData.data.xC) && ...
                                    isequal(obj.data_.yC, loadedData.data.yC) && ...
                                    isequal(obj.data_.xE, loadedData.data.xE) && ...
                                    isequal(obj.data_.yE, loadedData.data.yE);
                            end
                        else
                            if isfield(loadedData.data, 'x')
                                isConsistent = isequal(obj.data_.x, loadedData.data.x) && ...
                                    isequal(obj.data_.y, loadedData.data.y);
                            end
                        end
                    end

                    if isConsistent && isfield(obj.data_, 'dy2D')
                        if isfield(loadedData.data, 'dy2D')
                            isConsistent = isequal(obj.data_.dy2D, loadedData.data.dy2D);
                        else
                            isConsistent = false;
                        end
                    end
                    
                    if isConsistent
                        obj.data        = loadedData.data;
                        obj.hyperparams = loadedData.hyperparams;
                        obj.math        = loadedData.math;
                        
                        % Backward compatibility for models saved before normStats
                        if isfield(loadedData, "normStats")
                            obj.normStats = loadedData.normStats;
                        else
                            nOutLoad = length(loadedData.hyperparams);
                            nStatLoad = struct('mu', num2cell(zeros(1, nOutLoad)), 'sigma', num2cell(ones(1, nOutLoad)));
                            obj.normStats = nStatLoad;
                        end
                        
                        success = true;
                    else
                        fprintf("Dataset size or values changed. Retraining required.\n");
                    end
                else
                    warning("SurrogateModel:InvalidFile", "Invalid model file structure.");
                end
            end

            if ~success
                if isempty(fieldnames(varsIn))
                    error("SurrogateModel:NoSettings", "Training requires optimization settings.");
                end
                if options.autoTrain
                    obj = obj.fit('saveModel', options.saveModel);
                    return;
                end
                
                msg = sprintf("Model file missing or dataset changed:\n%s\n\nTraining may take time. Train now?", file);
                choice = questdlg(msg, "Missing Model", "Train", "Cancel", "Cancel");
                if strcmp(choice, "Train")
                    obj = obj.fit('saveModel', options.saveModel);
                else
                    error("SurrogateModel:Aborted", "Trained model required to continue.");
                end
            end
        end

        % -------------------------------------------------------------------------
        function obj = buildModelCHAD(obj)
            % Create Simulink-compatible surrogate representation.
            % Broadcasts params for fast execution. Seamlessly supports GEK
            % by zero-padding the augmented beta (psiV) arrays.
            if strcmpi(obj.surrogateType, "kriging") && ~isempty(obj.hyperparams_)
                nOut  = length(obj.hyperparams_);
                N     = size(obj.data_.x, 1);
                nDim  = size(obj.data_.x, 2);
                nPoly = size(obj.math_(1).mu, 1);
                
                % Find max length of beta to accommodate GEK augmented arrays
                maxBetaLen = max(arrayfun(@(m) length(m.beta), obj.math_));
                
                % Preallocate broadcasted arrays based on multi-output
                chad.x      = obj.data_.x;
                chad.mu     = zeros(nPoly, nOut);
                chad.theta  = zeros(N, nDim, nOut);
                chad.p      = zeros(N, nDim, nOut);
                chad.psiV   = zeros(maxBetaLen, nOut); % Padded to max length
                
                % Added Output Scaling Parameters
                chad.yMu    = zeros(1, nOut);
                chad.ySigma = zeros(1, nOut);
                
                % Added GEK Metadata for Simulink inference 
                chad.isGEK  = false(1, nOut);
                chad.dyDirs = zeros(nDim, nOut); % Padded index matrix
                
                for i = 1:nOut
                    chad.mu(:, i) = obj.math_(i).mu;
                    
                    % Extract standard p
                    pCurrent = obj.hyperparams_(i).p;
                    
                    % Handle GEK specific data
                    if isfield(obj.math_(i), 'isGEK') && obj.math_(i).isGEK
                        chad.isGEK(i) = true;
                        activeDirs = obj.math_(i).dyDirs;
                        
                        % Force mathematical consistency for derivatives
                        pCurrent(activeDirs) = 2; 
                        
                        % Store active directions padded with zeros
                        chad.dyDirs(1:length(activeDirs), i) = activeDirs(:);
                    end
                    
                    chad.theta(:, :, i) = repmat(obj.hyperparams_(i).theta, N, 1);
                    chad.p(:, :, i)     = repmat(pCurrent, N, 1);
                    
                    % Assign beta (padded with zeros if smaller than maxBetaLen)
                    beta_i = obj.math_(i).beta;
                    chad.psiV(1:length(beta_i), i) = beta_i;
                    
                    if ~isempty(obj.normStats_)
                        chad.yMu(i)    = obj.normStats_(i).mu;
                        chad.ySigma(i) = obj.normStats_(i).sigma;
                    end
                end
                
                obj.modelCHAD_ = chad;
            else
                % CoKriging or unsupported CHAD export
                obj.modelCHAD_ = struct();
            end
        end

        % -------------------------------------------------------------------------
        function ready = checkProperties(obj)
            % Validate internal consistency of data and outputs.
            ready = false;
            if isempty(fieldnames(obj.data_)) || isempty(obj.hyperparams_) || isempty(obj.math_) || isempty(obj.normStats_)
                return;
            end
            
            % Check data consistency
            if strcmpi(obj.surrogateType, "coKriging")
                if ~isfield(obj.data_, 'xE') || ~isfield(obj.data_, 'yE')
                    error("SurrogateModel:MissingData", "coKriging requires xE and yE.");
                end
                nOut = size(obj.data_.yE, 2);
            else
                if ~isfield(obj.data_, 'x') || ~isfield(obj.data_, 'y')
                    error("SurrogateModel:MissingData", "kriging requires x and y.");
                end
                nOut = size(obj.data_.y, 2);
            end
            
            % Check multi-output match
            if length(obj.hyperparams_) ~= nOut || length(obj.math_) ~= nOut || length(obj.normStats_) ~= nOut
                error("SurrogateModel:DimMismatch", ...
                    "Number of trained models (%d) does not match output dimensions (%d).", ...
                    length(obj.hyperparams_), nOut);
            end
            ready = true;
        end

        % -------------------------------------------------------------------------
        function obj = updateFcn(obj)
            % Dynamically resolve function handles based on surrogateType
            if isempty(obj.surrogateType) || strlength(obj.surrogateType) == 0
                return;
            end
            st = char(obj.surrogateType);
            obj.modelTuningFcn    = str2func(st + "Tuning");
            obj.modelPredictorFcn = str2func(st + "Predictor");
        end
    end

    %% ===================== SET / GET SYNC =====================
    methods
        function obj = set.modelName(obj, val), obj.modelName = val; end
        function obj = set.surrogateType(obj, val)
            obj.surrogateType = val;
            obj = obj.updateFcn();
        end
        function obj = set.data(obj, val)
            obj.data_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.data(obj), v = obj.data_; end
        
        function obj = set.hyperparams(obj, val)
            obj.hyperparams_ = val;
            obj.isReady = obj.checkProperties();
            if obj.isReady, obj = obj.buildModelCHAD(); end
        end
        function v = get.hyperparams(obj), v = obj.hyperparams_; end
        
        function obj = set.math(obj, val)
            obj.math_ = val;
            obj.isReady = obj.checkProperties();
            if obj.isReady, obj = obj.buildModelCHAD(); end
        end
        function v = get.math(obj), v = obj.math_; end

        function obj = set.normStats(obj, val)
            obj.normStats_ = val;
            obj.isReady = obj.checkProperties();
            if obj.isReady, obj = obj.buildModelCHAD(); end
        end
        function v = get.normStats(obj), v = obj.normStats_; end
        
        function obj = set.modelCHAD(obj, val), obj.modelCHAD_ = val; end
        function v = get.modelCHAD(obj)
            if strcmpi(obj.surrogateType, "coKriging")
                warning("modelCHAD is not fully supported for coKriging.");
            end
            v = obj.modelCHAD_;
        end
    end
end
