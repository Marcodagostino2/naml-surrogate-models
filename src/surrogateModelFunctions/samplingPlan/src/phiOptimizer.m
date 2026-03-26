function [bestPlan,bestPhi] = phiOptimizer(startPlan, population, ...
    maxIterations, metric, q)
% Evolutionary operation search for the most space filling Latin hypercube
% of a certain size and dimensionality. There is no need to call this
% directly - use bestLatinHypercube.m

n = size(startPlan,1);
initialPopulation = population;

bestPlan = startPlan;
bestPhi  = maxminPhi(bestPlan,metric,q);

leveloff = floor(0.85 * maxIterations);
bestPlanStreak = 0;
it = 0;
maxStreak = round(0.1 * maxIterations);

while (it < maxIterations && bestPlanStreak < maxStreak)

    it = it + 1;

    if it < leveloff
        mutations = round(1 + (0.5*n - 1) * (leveloff - it) / (leveloff - 1));
    else
        mutations = 1;
    end

    % Preallocate for parfor
    phiArray = inf(1, population);
    plans = cell(1, population);

    parfor offspring = 1: population
        planOff = perturb(bestPlan, mutations);
        phiOff  = maxminPhi(planOff, metric, q);

        phiArray(offspring) = phiOff;
        plans{offspring}    = planOff;
    end

    [improvedPhi, idxImp] = min(phiArray);
    improvedPlan = plans{idxImp};

    if improvedPhi < bestPhi
        bestPlan = improvedPlan;
        bestPhi  = improvedPhi;
        bestPlanStreak = 0;
        population = initialPopulation;
    else
        bestPlanStreak = bestPlanStreak + 1;
    end

    if bestPlanStreak == round(0.9 * maxStreak)
        population = 2 * initialPopulation;
    end
end

if it == maxIterations
    fprintf('Max iterations reached, streak = %d\n', bestPlanStreak);
else
    fprintf('Phi minimum found after %d iterations\n', it);
end

end
