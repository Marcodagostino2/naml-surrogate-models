classdef Settings < dynamicprops
    % Settings: Provides standardized way of loading config files
    %   Looks for files in missions > settings for global settings
    %   Looks for files in caller's folder otherwise
    %
    %   Constructor:
    %       - Settings: Creates an instance of the Settings class.
    %           Loaded config: -
    %           Loaded data: -
    %           Arguments:
    %               - configNames: config file names or path.
    %                   Accepts shortened version, e.g: ode -> odeConfig.m
    %
    %           If config name is provided, Settings checks inside the settings
    %           folder or inside the caller's folder

    properties(Constant, Access = protected)
        configPath = ''
        configName = ''
        variableName = ''
    end

    methods
        function obj = Settings(configNames)
            arguments (Input, Repeating)
                configNames      char
            end

            configNames = string(configNames);

            noConfig = ~endsWith(configNames, ["Config.m", "Config"]);
            noExtension = ~endsWith(configNames, ".m");

            configNames(noConfig) = strcat(configNames(noConfig), 'Config');
            configNames(noExtension) = strcat(configNames(noExtension), '.m');

            varsIn = obj.getConfig(configNames);
            obj.loadConfig(varsIn);
        end
    end

    methods(Static)
        function target = read(target, source, targetFields)
            arguments
                target      {mustBeA(target, {'struct', 'Settings'})}
                source      struct
            end

            arguments (Repeating)
                targetFields {mustBeA(targetFields, {'char', 'string'})}
            end
            % MERGE - reads and applies data from a source struct
            %   Usage: settings.read(<struct>, <name of settings field>)
            %
            %   Source fields have priority over settings fields, meaning
            %   the source will overwrite existing settings with the same
            %   name
            %
            %   Empty fields in the source struct are ignored

            fields = fieldnames(source)';
            if isempty(fields), return; end

            if length(targetFields) > 1
                target.(targetFields{1}) = Settings.read( ...
                    target.(targetFields{1}), ...
                    source, ...
                    targetFields{2:end});
                return;
            end

            % WIP: fix wrong assignment
            targetField = targetFields{1};
            for field = fields
                fd = field{1};
                if isempty(source.(fd)), continue; end
                target.(targetField).(fd) = source.(fd);
            end
        end
    end

    methods(Access = private)
        function loadConfig(obj, varsIn)
            props = fields(varsIn);
            for j = 1:length(props)
                obj.addprop(props{j});
                obj.(props{j}) = varsIn.(props{j});
            end
        end
    end

    methods(Static, Access = private)
        function outputVariables = getConfig(configNames)
            outputVariables = struct();
            for i = 1:length(configNames)
                fileName = configNames{i};
                filePath = '';

                % Look for config in caller function path
                if ~isfile(fullfile(filePath, fileName)) % Get caller function path
                    callers = dbstack("-completenames");
                    if length(callers) > 2
                        filePath = fileparts(callers(3).file);
                    end
                end

                if ~isfile(fullfile(filePath, fileName))
                    error(['File not found inside the target folder: %s, at\n' ...
                        '%s\n' ...
                        'Check that the config file exists'], fileName, filePath);
                end

                variables = [who(); "variables"];
                run(fileName);
                loadedVariables = who();
                loadedVariables(ismember(loadedVariables, variables)) = [];

                for j = 1:length(loadedVariables)
                    outputVariables.(loadedVariables{j}) = eval(loadedVariables{j});
                end
            end
        end
    end


    methods(Static, Access = private)
        function loadData(), end
    end
end
