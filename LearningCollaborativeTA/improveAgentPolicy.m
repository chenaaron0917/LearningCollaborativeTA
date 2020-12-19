clc;clear;close all
FRAMEWORK_ITER = 2;
rng(FRAMEWORK_ITER)
load initTrain.mat
filename = char("humanPolicy"+num2str(FRAMEWORK_ITER));
load(filename)
filename = char("prePolicyWalk"+num2str(FRAMEWORK_ITER));
load(filename)

%% Environment 
% environment
env = OperationEnv(env_param);
env.reset;
env_param.agent_capa


gamma = 0.95;
learner = learning(env, gamma, human_reward_w', agents_pi, T_pi);
[human_pi, human_Q] = learner.policyIteration();
%% Q learning for agents
% algo parameters

gamma = 0.95; % discount
alpha = 0.005; % Q-learning rate
max_episode = 2000; % training episode
epsilon = 1; % random action
epsilon_decay = 0.9965;
min_epsilon = 0.001;


% start Decentralized Q-Learning
track_reward = zeros(1,max_episode);
for episode = 1:max_episode
    env.reset;
    op_state = env.op_state;
    episode_reward = 0;
    done = 0;
    while done ~= 1
        if rand > epsilon % optimal assignment
            assign = zeros(env.n_agents-1,1);
            for ag = 1:env.n_agents-1
                Q = allagent_Qs{ag};
                [maxQ, opt_assign] = max(Q(op_state,:));
                assign(ag) = opt_assign;
            end
        else % random assignment
            assign = randi([1,env.n_tasks],[env.n_agents-1,1]);
        end
        assign(end+1) = human_pi(op_state);
        % evolve state
        assign = reshape(assign, env.n_agents, 1);
        [~, next_op_state, step, done] = env.step(assign);
%         next_op_state
        % get agent's reward based on state
        agent_rewards = zeros(1,env.n_agents-1);
        for ag = 1:env.n_agents-1
            w = agent_reward_W{ag};
            [aReward, level] = getAgentReward(env.demand_state, w);
            agent_rewards(ag) = aReward;
        end
        if next_op_state == 1
            reward = 10;
        else
            reward = 0;
        end
%         reward = mean(agent_rewards);
        episode_reward = episode_reward + (gamma^step) * reward;
        % update individual Q tables
        for ag = 1:env.n_agents-1
            Qs = allagent_Qs{ag}(op_state,assign(ag));
            bestQs_next = max(allagent_Qs{ag}(next_op_state,:));
            Qs_update = (1-alpha) * Qs + alpha * (reward + gamma * bestQs_next);
            allagent_Qs{ag}(op_state,assign(ag)) = Qs_update;
        end
        op_state = next_op_state;
    end
    % info
    if mod(episode,100) == 0
        disp(strcat(num2str(episode),", ",num2str(episode_reward)))
    end
    % random action decay
    if epsilon > min_epsilon
        epsilon = epsilon*epsilon_decay;
        epsilon = max(min_epsilon, epsilon);
    end
    track_reward(episode) = episode_reward;
end

%% Plots
stats_every = 20;
n_stats = length(track_reward)/stats_every;
stats_x = 1:n_stats;
stats_mean = zeros(1,n_stats);
stats_std = zeros(1,n_stats);
for i=1:n_stats
    sub_track_reward = track_reward((i-1)*stats_every+1:i*stats_every);
    stats_mean(i)= mean(sub_track_reward);
    stats_std(i) = std(sub_track_reward);
end
figure()
plot(stats_x,stats_mean,'b','LineWidth',2)
hold on
p1 = patch([stats_x fliplr(stats_x)], [stats_mean+stats_std fliplr(stats_mean-stats_std)], 'b','FaceAlpha',0.2);
plot(stats_x,stats_mean+stats_std,'b','LineWidth',0.5)
plot(stats_x,stats_mean-stats_std,'b','LineWidth',0.5)
ticks = 0:stats_every:n_stats;
xticks(ticks)
xticklabels(ticks.*stats_every)

%%
filename = char("improvedAgent"+num2str(FRAMEWORK_ITER));
save(filename, 'allagent_Qs', 'human_reward_w')


%% Functions
function [reward, level] = getAgentReward(state,w)
    level = sum(state) - length(state) + 1;
    reward = w(level);
end