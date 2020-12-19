clc;clear;close all
FRAMEWORK_ITER = 2;
rng(FRAMEWORK_ITER)
%% Start Policy Walk
% load initTrain.mat
filename = char("prePolicyWalk"+num2str(FRAMEWORK_ITER));
load(filename)
% PW parameters
max_iter = 100;
step_size = 1;
r_max = 5;
alpha = 0.1;

%% Step 1
disp('Start Policy Walk')
if FRAMEWORK_ITER >= 1
    x = 1:env.n_tasks*(env.n_task_levels-1)+1;
    human_reward_w = sampleRandomRewards(length(x), step_size, r_max);
else
    human_reward_w = human_reward_w';
end
    % human_reward_w
    % learner class
gamma = 0.95;
learner = learning(env, gamma, human_reward_w, agents_pi, T_pi);

%% Step 2
    [human_pi, human_Q] = learner.policyIteration();

%% Step 3
samples = [];
for iter = 1:max_iter
    iter
    learner_cand = learner;
%     a = learner.human_reward_w
    learner_cand.human_reward_w = mcmcRewardStep(learner.human_reward_w, step_size, r_max);
    learner_cand.human_reward = learner_cand.human_reward_w(learner_cand.env.getOpLevel(learner_cand.env.op_states));
    Q_human_pi_cand = learner_cand.getQforPi(human_pi);
    if existsBetterSolution(human_pi, Q_human_pi_cand)
        [human_pi_cand, human_Q_cand] = learner_cand.policyIteration();
        ratio = getPosteriorRatio(demo, learner_cand, human_pi_cand, learner, human_pi, r_max, alpha);
        if rand < ratio
            learner = learner_cand;
            human_pi = human_pi_cand;
        end
    else
        ratio = getPosteriorRatio(demo, learner_cand, human_pi, learner, human_pi, r_max, alpha);
        if rand < ratio
            learner = learner_cand;
        end
    end
    ratio
%     learner.human_reward_w'
    samples = [samples; learner.human_reward_w'];
end

%% save human policy data
human_reward_w = mode(samples);
filename = char("humanPolicy"+num2str(FRAMEWORK_ITER));
save(filename, 'human_pi', 'samples', 'human_reward_w')
% save humanPolicy.mat human_pi samples


%% Functions
% reward based on sum of the task demand levels
function level = getStateLevel(state)
    level = sum(state) - length(state) + 1;
end

% sampling a random reward based on a grid with step_size
function rewards = sampleRandomRewards(n_states, step_size, r_max)
    % 0 - r_max uniform + stepsize
    rewards = (r_max).*rand(n_states,1) + step_size;
%     rewards = (2*r_max).*rand(n_states,1) + step_size;
    % fix on grid
    reward_mod = mod(rewards, step_size);
%     rewards = rewards - reward_mod - r_max - step_size;
    rewards = rewards - reward_mod - step_size;
end

% MCMC step for sampling reward
function rewards_cand = mcmcRewardStep(rewards, step_size, r_max)
    rewards_cand = rewards;
    idx = randi(length(rewards));
    step = randsample([-step_size, step_size],1);
    rewards_cand(idx) = rewards_cand(idx) + step;
    rewards_cand = max(min(rewards_cand, r_max), 0);
    if isequal(rewards_cand, rewards)
        rewards_cand(idx) = rewards_cand(idx) - step;
    end
end

% check if there exists better Q values
function exist = existsBetterSolution(human_pi, Q_human_pi)
    exist = 0;
    [n_states, n_actions] = size(Q_human_pi);
    for s = 1:n_states
        for a = 1:n_actions
            if Q_human_pi(s, human_pi(s)) < Q_human_pi(s, a)
                exist = 1;
                s = n_states;
                break
            end
        end
    end
end

function ratio = getPosteriorRatio(demos, learner_cand, pi_cand, learner, pi, r_max, alpha)
    log_p_cand = getLogPosterior(demos, learner_cand, pi_cand, r_max, alpha);
    log_p = getLogPosterior(demos, learner, pi, r_max, alpha);
    ratio = exp(log_p_cand - log_p);
end

function log_p = getLogPosterior(demos, learner, pi, r_max, alpha)
    q = learner.getQforPi(pi);
    log_p = 0;
    for d_i = 1:length(demos)
        s = demos(d_i,1);
        a = demos(d_i,2);
        log_p = log_p + (alpha*q(s,a) - logsumexp(alpha*q(s,:), 2));
    end
    % uniform log prior
%     log_prior = learner.getLogPrior(r_max, 'uniform');
    % beta log prior
    log_prior = learner.getLogPrior(r_max, 'beta');
%     learner.human_reward'
    log_p = log_p + log_prior;
end