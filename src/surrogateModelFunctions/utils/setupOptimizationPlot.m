function outfun = setupOptimizationPlot(figTitle, d)
% setupOptimizationPlot  Create a live plotting callback for fmincon.
%
% Generates a figure and returns an OutputFcn callback for the optimizer.
% History state is safely encapsulated in the figure's UserData property 
% rather than using nested closures (no global variables).

    figName = 'Surrogate Optimization Progress';
    fig = findobj('Type', 'Figure', 'Name', figName);
    
    if isempty(fig)
        fig = figure('Name', figName, 'Color', 'w');
    else
        clf(fig); 
    end
    
    sgtitle(figTitle, 'Interpreter', 'latex');
    set(fig, 'defaultTextInterpreter', 'latex');
    set(fig, 'defaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'defaultLegendInterpreter', 'latex');
    
    stateData.loss   = [];
    stateData.grad   = [];
    stateData.theta  = [];
    stateData.p      = [];
    stateData.lambda = [];
    stateData.startCount = 0;
    stateData.d = d;
    stateData.thetaLabels = arrayfun(@(i) sprintf('$$\\theta_%d$$', i), 1:d, 'UniformOutput', false);
    stateData.pLabels     = arrayfun(@(i) sprintf('$$p_%d$$', i), 1:d, 'UniformOutput', false);
    
    fig.UserData = stateData;
    
    outfun = @(z, optimValues, state) plotCallback(z, optimValues, state, fig);
end

function stop = plotCallback(z, optimValues, state, fig)
    stop = false;
    
    if ~isvalid(fig), return; end
    
    stateData = fig.UserData;
    d = stateData.d;
    
    switch state
        case 'init'
            stateData.startCount = stateData.startCount + 1;
            stateData.loss   = [];
            stateData.grad   = [];
            stateData.theta  = [];
            stateData.p      = [];
            stateData.lambda = [];
            
            fig.UserData = stateData; 
            return;
            
        case 'iter'
            stateData.loss   = [stateData.loss; optimValues.fval];
            stateData.theta  = [stateData.theta; 10.^z(1:d)];
            stateData.p      = [stateData.p; z(d+1:2*d)];
            stateData.lambda = [stateData.lambda; z(end)];
            
            gNorm = optimValues.firstorderopt;
            if isempty(gNorm), gNorm = NaN; end
            stateData.grad = [stateData.grad; gNorm];
            
            fig.UserData = stateData;
    end
    
    iters = 1:size(stateData.loss, 1);
    if isempty(iters), return; end
    
    set(0, 'CurrentFigure', fig);
    
    subplot(2, 3, 1);
    plot(iters, stateData.loss, 'LineWidth', 1.5, 'Color', '#0072BD');
    title(sprintf('Loss (Start %d)', stateData.startCount), 'Interpreter', 'latex');
    xlabel('Iteration', 'Interpreter', 'latex');
    ylabel('$$\mathcal{L}(\mathbf{z})$$', 'Interpreter', 'latex');
    grid on;
    
    subplot(2, 3, 2);
    semilogy(iters, stateData.grad, 'LineWidth', 1.5, 'Color', '#D95319');
    title('1st-Order Optimality', 'Interpreter', 'latex');
    xlabel('Iteration', 'Interpreter', 'latex');
    ylabel('$$\| \nabla \mathcal{L} \|_{\infty}$$', 'Interpreter', 'latex');
    grid on;
    
    subplot(2, 3, 3);
    semilogy(iters, stateData.theta, 'LineWidth', 1.5);
    title('$$\theta$$ Parameters', 'Interpreter', 'latex');
    xlabel('Iteration', 'Interpreter', 'latex');
    ylabel('$$\theta_i$$', 'Interpreter', 'latex');
    legend(stateData.thetaLabels, 'Location', 'best', 'Interpreter', 'latex');
    grid on;
    
    subplot(2, 3, 4);
    plot(iters, stateData.p, 'LineWidth', 1.5);
    title('$$p$$ Parameters', 'Interpreter', 'latex');
    xlabel('Iteration', 'Interpreter', 'latex');
    ylabel('$$p_i$$', 'Interpreter', 'latex');
    ylim([0.9, 2.1]);
    legend(stateData.pLabels, 'Location', 'best', 'Interpreter', 'latex');
    grid on;
    
    subplot(2, 3, 5);
    semilogy(iters, stateData.lambda, 'LineWidth', 1.5, 'Color', '#7E2F8E');
    title('$$\lambda$$ (Nugget)', 'Interpreter', 'latex');
    xlabel('Iteration', 'Interpreter', 'latex');
    ylabel('$$\lambda$$', 'Interpreter', 'latex');
    grid on;
    
    drawnow;
end
