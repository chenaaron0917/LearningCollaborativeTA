clc;clear;close all
rng(1);
%% Environment Parameters
% agents and tasks
env_param.n_agents = 3;
% env_param.agent_capa = 2*ones(2,env_param.n_agents);
env_param.agent_capa = [1,1,1;1,1,1];
env_param.n_tasks = 4;
env_param.n_task_levels = 3;
env_param.op_states = sortrows(combinator(env_param.n_task_levels,env_param.n_tasks,'p','r'));
env_param.n_op_states = size(env_param.op_states, 1);
env_param.actions = sortrows(combinator(env_param.n_tasks,env_param.n_agents,'p','r'));
env_param.n_actions = size(env_param.actions, 1);

env_param.n_capa_levels = 3;
% task transistion probabilities
env_param.fireT = fireTransProb(env_param.n_task_levels,env_param.n_capa_levels);
env_param.rescueT = rescueTransProb(env_param.n_task_levels,env_param.n_capa_levels);


%% Play with the env
% environment
env = OperationEnv(env_param);
env.reset;
% TT = env.fireT(2,:,:);
% TT2 = squeeze(TT);
% rTT = env.rescueT(2,1,:,:);
% rTT2 = squeeze(rTT);

% env.episode_step
% [reward,done] = env.step([1,1,1]');
% env.demand_state
% env.episode_step

%% Q Learning for three agents

% agents
% individual reward weight
agent_reward_W = {};
for ag = 1:env.n_agents
    % reward weight 1
%     x = 1:env.n_tasks*(env.n_task_levels-1)+1;
%     x = x.^3;
%     w = round(x/x(end),2);
    % reward weight 2
    w = zeros(env.n_tasks*(env.n_task_levels-1)+1,1);
    w(1) = 10;
    agent_reward_W{ag} = w;
end
% algo parameters

gamma = 0.95; % discount
alpha = 0.005; % Q-learning rate
max_episode = 2000; % training episode
epsilon = 1; % random action
epsilon_decay = 0.9965;
min_epsilon = 0.001;


% Q table for each agent
% allagent_Qs = {};
% for ag = 1:env.n_agents
%     Qdim = env.n_task_levels*ones(1,env.n_tasks);
%     Qdim(end+1) = env.n_tasks;
%     allagent_Qs{ag} = 1*ones(Qdim);
% end

allagent_Qs = {};
for ag = 1:env.n_agents
    allagent_Qs{ag} = 1*ones(env.n_op_states, env.n_tasks);
end

% start Decentralized Q-Learning
track_reward = zeros(1,max_episode);
for episode = 1:max_episode
    env.reset;
    op_state = env.op_state;
    episode_reward = 0;
    done = 0;
    while done ~= 1
        if rand > epsilon % optimal assignment
            assign = zeros(env.n_agents,1);
            for ag = 1:env.n_agents
                Q = allagent_Qs{ag};
                [maxQ, opt_assign] = max(Q(op_state,:));
                assign(ag) = opt_assign;
            end
        else % random assignment
            assign = randi([1,env.n_tasks],[env.n_agents,1]);
        end
        % evolve state

        [reward, next_op_state, step, done] = env.step(assign);
%         next_op_state
        % get agent's reward based on state
        agent_rewards = zeros(1,env.n_agents);
        for ag = 1:env.n_agents
            w = agent_reward_W{ag};
            [aReward, level] = getAgentReward(env.demand_state, w);
            agent_rewards(ag) = aReward;
        end

%         reward = mean(agent_rewards);
        episode_reward = episode_reward + (gamma^step) * reward;
        % update individual Q tables
        for ag = 1:env.n_agents
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

save initTrain.mat env_param allagent_Qs agent_reward_W
%% Functions
function [reward, level] = getAgentReward(state,w)
    level = sum(state) - length(state) + 1;
    reward = w(level);
end

