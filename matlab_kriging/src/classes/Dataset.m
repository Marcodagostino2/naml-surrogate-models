classdef Dataset 
    % Dataset
    % -------------------------------------------------------------------------
    % Encapsulates the full lifecycle of a dataset used in surrogate modeling
    % workflows: loading, building, splitting, and dimensional consistency
    % checking.
    %
    % Supports dynamic builder resolution based on modelName:
    %   "apogee"      → buildApogeeDataset
    %   "datcom"      → buildDatcomDataset
    %   "cfd"         → buildCfdDataset
    %
    % Design goals:
    %   • Lazy evaluation: data is only checked/resolved when assigned
    %   • Automatic dimensional consistency via setter-triggered checkProperties
    %   • Interactive or automated build workflows 
    %   • Clean value-class semantics: all mutations return a new object
    %
    % -------------------------------------------------------------------------

    %% ===================== PUBLIC IMMUTABLE PROPERTIES =====================
    properties (SetAccess = private)

        % Logical dataset identifier ("apogee", "datcom", ...)
        % Drives dynamic resolution of the builder via updateFcn
        modelName  (1,1) string

        % Options struct forwarded to the builder function during generation
        datasetOptions struct

        % Function handle to dataset builder, resolved from modelName:
        %   modelName "foo" → str2func("buildFooDataset")
        builder function_handle

        % Readiness flag: TRUE when data is loaded and dimensionally consistent
        isReady logical = false
    end

    properties (SetAccess = private)
        splitPerc (1,1) double        % Split percentage
        trainIdx  (:,1) logical       % Logical indeces for training
        testIdx   (:,1) logical       % Logical indeces for testing
    end

    %% ===================== CONSTANT PRIVATE PROPERTIES =====================
    properties (Constant, Access = private)
        projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))))
        configPath  = fullfile(Dataset.projectRoot, 'config')
        dataPath    = fullfile(Dataset.projectRoot, 'data')
    end

    %% ===================== DEPENDENT PUBLIC PROPERTIES =====================
    % Backed by private storage to enable setter-triggered side effects
    % (dimensional consistency checks on every assignment)
    properties (Dependent, SetAccess = private)
        % Input coordinates in unit hypercube — [N x nInputs]
        xUnit
        % Input coordinates in physical space — [N x nInputs]
        % Optional: only populated when physical coordinates differ from unit
        xReal
        % Output samples — [N x nOutputs]
        y
        % 3D Gradient matrix [nPoints x nOutputs x nInputs] (Generated on the fly)
        dy          
        % 2D Dense gradient matrix [nPoints x nGrads] (Memory efficient)
        dy2D       
        % Physical input ranges — [2 x nInputs], row 1: min, row 2: max
        ranges
        % Power-transform exponents applied before unit normalization — [1 x nInputs]
        power
        % Provenance metadata: geometry, variable names, data hash
        metadata
    end

    %% ===================== PRIVATE BACKING STORAGE =====================
    properties (Access = private)
        xUnit_    (:,:) double
        xReal_    (:,:) double
        y_        (:,:) double
        dy2D_     (:,:) double   
        ranges_   (2,:) double
        power_    (1,:) double
        metadata_ (1,1) struct
    end

    %% ===================== CONSTRUCTOR =====================
    methods
        function obj = Dataset(modelName, configPath, varsIn, options)
            % Constructor
            % -------------------------------------------------------------------------
            % Loads or builds the dataset depending on available files.
            %
            % INPUTS
            %   modelName   → string identifier for dynamic builder resolution
            %   configPath  → path to config file; used only when varsIn is empty
            %   varsIn  → options struct for generation/loading
            %                 must contain at least varsIn.savePath
            %
            % OPTIONS (name-value via loadOptions)
            %   autoBuild → if true, build without interactive prompt
            %   saveDataset → persist dataset to file after building
            %
            % Lifecycle:
            %   1. Assign modelName → triggers updateFcn (resolves datasetFcn)
            %   2. Resolve varsIn from configPath if not provided
            %   3. Attempt file load; fall back to build (auto or interactive)
            %   4. Final readiness validation
            arguments  
                modelName         (1,1) string 
                configPath        (1,1) string    = ""
                varsIn                  struct  = []
                options.reBuild         logical = false
                options.autoBuild       logical = false
                options.saveDataset     logical = true
            end

            % Assignment via setter → triggers updateFcn → resolves datasetFcn
            obj.modelName = modelName;

            % Store only if non-empty; do not assign empty default so that
            % missing varsIn is detectable downstream (errors on build)
            if ~isempty(fieldnames(varsIn))
                obj.datasetOptions = varsIn;
            else
                % Autoresolve configPath if not explicitly provided
                if strlength(configPath) == 0
                    configPath = fullfile(obj.configPath, "datasets", ...
                        obj.modelName + "DatasetConfig");
                end
                
                % Load settings from the resolved path
                dataSettings = Settings(configPath);
                obj.datasetOptions = dataSettings.datasetOptions;
            end

            obj = obj.loadDataset(...
                'autoBuild', options.autoBuild, ...
                'saveDataset', options.saveDataset, ...
                'reBuild', options.reBuild);

            % Explicit final check (redundant but documents intent)
            obj.isReady = obj.checkProperties();
        end
    end

    %% ===================== PUBLIC API =====================
    methods (Access = public)

        function [obj, trainSet, testSet] = split(obj, perc)
            arguments
                obj
                perc = [];
            end

            % Check or recompute split indices
            if isempty(perc)
                if isempty(obj.trainIdx)
                    error("Dataset:NoSplit", "You need to specify the split percentage if no one is saved.");
                end
            else
                nData = size(obj.y, 1);
                % Recompute indices if missed or changed
                if isempty(obj.trainIdx) || abs(obj.splitPerc - perc) > 1e-6
                    obj.splitPerc = perc;
                    % rng(42, 'twister');
                    idx = randperm(nData);
                    nTrain = round(perc * nData);

                    % Logical mask creation
                    obj.trainIdx = false(nData, 1);
                    obj.trainIdx(idx(1:nTrain)) = true;
                    obj.testIdx = ~obj.trainIdx;

                    obj.saveDataset();
                end
            end

            % Extract standard subsets using the logical masks
            trainSet.xUnit = obj.xUnit(obj.trainIdx, :);
            trainSet.xReal = obj.xReal(obj.trainIdx, :);
            trainSet.y     = obj.y(obj.trainIdx, :);
            testSet.xUnit  = obj.xUnit(obj.testIdx, :);
            testSet.xReal  = obj.xReal(obj.testIdx, :);
            testSet.y      = obj.y(obj.testIdx, :);
            
            if ~isempty(obj.dy2D_)
                trainSet.dy2D = obj.dy2D(obj.trainIdx, :);
                testSet.dy2D  = obj.dy2D(obj.testIdx, :);
            end
        end

        % -------------------------------------------------------------------------
        function plot(obj, options)
            arguments
                obj
                options.exportPath   (1,1) string  = ""
                options.exportFormat (1,1) string  = "png"
                options.maxPairs     (1,1) double  = 6
                options.fontSize     (1,1) double  = 11
            end
            p = DatasetPlotter(obj, ...
                "exportPath",   options.exportPath,   ...
                "exportFormat", options.exportFormat, ...
                "maxPairs",     options.maxPairs,     ...
                "fontSize",     options.fontSize);
            p.plotAll();
        end

        % -------------------------------------------------------------------------
        function obj = build(obj, options)
            % Generate dataset by calling the dynamically resolved builder.
            %
            % The builder must return a struct with at least xUnit and y.
            % Optional fields: xReal, ranges, power, metadata.
            arguments
                obj                 Dataset
                options.saveDataset logical = true
            end
            if isempty(obj.builder)
                error("Dataset:NoBuilder", ...
                    "datasetFcn is not resolved. Assign modelName first.");
            end
            fprintf("Building %s dataset...\n", obj.modelName);
            raw = obj.builder(obj.datasetOptions);
            obj = obj.assignDataFields(raw);
            fprintf("Build complete: %d samples, %d inputs, %d outputs.\n", ...
                size(obj.xUnit_, 1), size(obj.xUnit_, 2), size(obj.y_, 2));
            if options.saveDataset
                obj.saveDataset();
            end
        end

        % -------------------------------------------------------------------------
        function saveDataset(obj)
            file = fullfile(obj.dataPath, "datasets", ...
                obj.modelName + "Dataset.mat");

            data.xUnit     = obj.xUnit_;
            data.xReal     = obj.xReal_;
            data.y         = obj.y_;
            data.dy2D      = obj.dy2D_;
            data.ranges    = obj.ranges_;
            data.power     = obj.power_;
            data.metadata  = obj.metadata_;
            data.splitPerc = obj.splitPerc;
            data.trainIdx  = obj.trainIdx;
            data.testIdx   = obj.testIdx;

            save(file, "-struct", "data");
            fprintf("Dataset saved → %s\n", file);

        end
    end

    %% ===================== PRIVATE LIFECYCLE =====================
    methods (Access = private)

        function obj = loadDataset(obj, options)
            % Attempt to load dataset from file; fall back to building it.
            %
            % File path is resolved from varsIn.savePath.
            % If missing: autoBuild triggers silent build, otherwise
            % an interactive dialog prompts the user.
            arguments
                obj                 Dataset
                options.autoBuild   logical
                options.saveDataset logical
                options.reBuild     logical
            end

            varsIn = obj.datasetOptions;
            
            if options.reBuild
                obj = obj.build(...
                    'saveDataset', options.saveDataset);
                return;
            end

            file = fullfile(obj.dataPath, "datasets", ...
                obj.modelName + "Dataset.mat");

            success = false;
            if ~isempty(file) && exist(file, 'file')
                raw = load(file);
                obj = obj.assignDataFields(raw);
                success = true;
            end

            if ~success
                if isempty(fieldnames(varsIn))
                    error("Dataset:NoOptions", ...
                        "datasetOptions is required to build the dataset.");
                end
                if options.autoBuild
                    obj = obj.build(...
                        'saveDataset', options.saveDataset);
                    return;
                end
                % Interactive fallback
                msg = sprintf("Dataset file missing or invalid:\n%s\n\nGenerate now?", file);
                choice = questdlg(msg, "Missing Dataset", "Generate", "Cancel", "Cancel");
                if strcmp(choice, "Generate")
                    obj = obj.build(...
                        'saveDataset', options.saveDataset);
                else
                    error("Dataset:Aborted", "Dataset required to continue.");
                end
            end
        end

        % -------------------------------------------------------------------------
        function obj = assignDataFields(obj, raw)
            % Assign fields from a raw struct to object properties.
            obj.xUnit = raw.xUnit;
            obj.y     = raw.y;
            
            if isfield(raw, 'xReal'),    obj.xReal    = raw.xReal;    end
            if isfield(raw, 'dy2D'),     obj.dy2D = raw.dy2D;
            elseif isfield(raw, 'dy'),   obj.dy2D = raw.dy;           end
            if isfield(raw, 'ranges'),   obj.ranges   = raw.ranges;   end
            if isfield(raw, 'power'),    obj.power    = raw.power;    end
            if isfield(raw, 'metadata'), obj.metadata = raw.metadata; end
            
            % Restore split states if they exist
            if isfield(raw, 'splitPerc'), obj.splitPerc = raw.splitPerc; end
            if isfield(raw, 'trainIdx'),  obj.trainIdx  = raw.trainIdx;  end
            if isfield(raw, 'testIdx'),   obj.testIdx   = raw.testIdx;   end
        end

        % -------------------------------------------------------------------------
        function ready = checkProperties(obj)
            % Validate internal dimensional consistency across all stored fields.
            %
            % Rules enforced:
            %   [1] xUnit and y must have the same number of rows
            %   [2] ranges (if set) must have nInputs columns
            %   [3] power  (if set) must have nInputs elements
            %   [4] xReal  (if set) must match xUnit shape exactly
            if isempty(obj.xUnit_) || isempty(obj.y_)
                ready = false;
                return;
            end
            nSamples = size(obj.xUnit_, 1);
            nInputs  = size(obj.xUnit_, 2);

            if size(obj.y_, 1) ~= nSamples
                error("Dataset:DimMismatch", ...
                    "xUnit has %d rows but y has %d rows.", ...
                    nSamples, size(obj.y_, 1));
            end
            if ~isempty(obj.dy2D_)
                if size(obj.dy2D_, 1) ~= nSamples
                    error("Dataset:DimMismatch", "dy2D has %d rows but xUnit has %d.", size(obj.dy2D_, 1), nSamples);
                end
            end
            if ~isempty(obj.ranges_) && size(obj.ranges_, 2) ~= nInputs
                error("Dataset:RangesMismatch", ...
                    "ranges has %d columns but xUnit has %d inputs.", ...
                    size(obj.ranges_, 2), nInputs);
            end
            if ~isempty(obj.power_) && numel(obj.power_) ~= nInputs
                error("Dataset:PowerMismatch", ...
                    "power has %d elements but xUnit has %d inputs.", ...
                    numel(obj.power_), nInputs);
            end
            if ~isempty(obj.xReal_)
                if size(obj.xReal_, 1) ~= nSamples || size(obj.xReal_, 2) ~= nInputs
                    error("Dataset:xRealMismatch", ...
                        "xReal [%dx%d] does not match xUnit [%dx%d].", ...
                        size(obj.xReal_, 1), size(obj.xReal_, 2), nSamples, nInputs);
                end
            end
            ready = true;
        end

        % -------------------------------------------------------------------------
        function obj = updateFcn(obj)
            % Resolve datasetFcn from modelName using naming convention:
            %   modelName "foo" → str2func("buildFooDataset")
            %
            % Called automatically by set.modelName.
            if isempty(obj.modelName) || strlength(obj.modelName) == 0
                return;
            end
            cap = obj.capitalize(obj.modelName);
            obj.builder = str2func("build" + cap + "Dataset");
        end
    end

    %% ===================== STATIC UTILITIES =====================
    methods (Static, Access = private)
        function out = capitalize(str)
            str = char(str);
            out = string([upper(str(1)) str(2:end)]);
        end
        % -------------------------------------------------------------------------
        function valid = isValidRaw(raw)
            % Check that a loaded struct contains the minimum required fields.
            valid = isstruct(raw) && isfield(raw, 'xUnit') && isfield(raw, 'y');
        end
    end

    %% ===================== SET / GET SYNC =====================
    methods
        % --- modelName: triggers updateFcn on every assignment ---
        function obj = set.modelName(obj, val)
            obj.modelName = val;
            obj = obj.updateFcn();
        end

        % --- xUnit: consistency check on assignment ---
        function obj = set.xUnit(obj, val)
            obj.xUnit_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.xUnit(obj),  v = obj.xUnit_;    end

        % --- xReal: consistency check on assignment ---
        function obj = set.xReal(obj, val)
            obj.xReal_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.xReal(obj),  v = obj.xReal_;    end

        % --- y: consistency check on assignment ---
        function obj = set.y(obj, val)
            obj.y_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.y(obj),      v = obj.y_;        end

        % --- dy2D: Dense 2D gradient storage ---
        function obj = set.dy2D(obj, val)
            obj.dy2D_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.dy2D(obj)
            v = obj.dy2D_;
        end

        % --- dy: On-the-fly 3D Matrix construction (Read-Only) ---
        function v = get.dy(obj)
            if isempty(obj.dy2D_)
                v = [];
                return;
            end

            [nPoints, nGrads] = size(obj.dy2D_);
            nOut = size(obj.y_, 2);
            nIn  = size(obj.xUnit_, 2);

            v = NaN(nPoints, nOut, nIn);

            if isfield(obj.metadata_, 'gradMap')
                gradMap = obj.metadata_.gradMap;
                for g = 1:nGrads
                    outIdx = gradMap(g, 1);
                    inIdx  = gradMap(g, 2);
                    if ~isnan(outIdx) && ~isnan(inIdx)
                        v(:, outIdx, inIdx) = obj.dy2D_(:, g);
                    end
                end
            end
        end

        % --- ranges: consistency check on assignment ---
        function obj = set.ranges(obj, val)
            obj.ranges_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.ranges(obj), v = obj.ranges_;   end

        % --- power: consistency check on assignment ---
        function obj = set.power(obj, val)
            obj.power_ = val;
            obj.isReady = obj.checkProperties();
        end
        function v = get.power(obj),  v = obj.power_;    end

        % --- metadata: no dimensional side effects ---
        function obj = set.metadata(obj, val)
            obj.metadata_ = val;
        end
        function v = get.metadata(obj), v = obj.metadata_; end
    end
end
