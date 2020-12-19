clc;clear;close all
rng(1);
load initTrain.mat
init_allagent_Qs = allagent_Qs;
init_rewrad_W = agent_reward_W;
load humanPolicy2.mat
load prePolicyWalk2.mat
%%
gamma = 0.95;
learner = learning(env, gamma, human_reward_w', agents_pi, T_pi);
[human_pi, human_Q] = learner.policyIteration();

load improvedAgent1.mat
iter1_allagent_Qs = allagent_Qs;
load improvedAgent2.mat
iter2_allagent_Qs = allagent_Qs;
%% Evaluation
% evaluation parameters
n_eval = 2000;
gamma = 0.95;

% environment
env = OperationEnv(env_param);
env.reset;

% start evaluation
track_score = zeros(1,n_eval);
for eval_i = 1:n_eval
    env.reset;
    op_state = env.op_state;
    eval_score = 0;
    done = 0;
    while done ~= 1
        assign = zeros(env.n_agents-1,1);
        for ag = 1:env.n_agents-1
            Q = init_allagent_Qs{ag};
            [maxQ, opt_assign] = max(Q(op_state,:));
            assign(ag) = opt_assign;
        end
        assign(end+1) = human_pi(op_state);
        assign = reshape(assign, env.n_agents, 1);
        % evolve state
        [env_reward, next_op_state, step, done] = env.step(assign);

        % get reward from environment
        eval_score = eval_score + (gamma^step) * env_reward;

        op_state = next_op_state;
    end
    track_score(eval_i) = eval_score;
end


%% evaluation perf after improving agent policy
% start evaluation
track_score2 = zeros(1,n_eval);
for eval_i = 1:n_eval
    env.reset;
    op_state = env.op_state;
    eval_score = 0;
    done = 0;
    while done ~= 1
        assign = zeros(env.n_agents-1,1);
        for ag = 1:env.n_agents-1
            Q = iter1_allagent_Qs{ag};
            [maxQ, opt_assign] = max(Q(op_state,:));
            assign(ag) = opt_assign;
        end
        assign(end+1) = human_pi(op_state);
        assign = reshape(assign, env.n_agents, 1);
        % evolve state
        [env_reward, next_op_state, step, done] = env.step(assign);

        % get reward from environment
        eval_score = eval_score + (gamma^step) * env_reward;

        op_state = next_op_state;
    end
    track_score2(eval_i) = eval_score;
end

%% evaluation perf after improving agent policy
% start evaluation
track_score3 = zeros(1,n_eval);
for eval_i = 1:n_eval
    env.reset;
    op_state = env.op_state;
    eval_score = 0;
    done = 0;
    while done ~= 1
        assign = zeros(env.n_agents-1,1);
        for ag = 1:env.n_agents-1
            Q = iter2_allagent_Qs{ag};
            [maxQ, opt_assign] = max(Q(op_state,:));
            assign(ag) = opt_assign;
        end
        assign(end+1) = human_pi(op_state);
        assign = reshape(assign, env.n_agents, 1);
        % evolve state
        [env_reward, next_op_state, step, done] = env.step(assign);

        % get reward from environment
        eval_score = eval_score + (gamma^step) * env_reward;

        op_state = next_op_state;
    end
    track_score3(eval_i) = eval_score;
end
%%
figure(1)
subplot(1,3,1)
histogram(track_score,7)
xlim([-5,130])
ylim([0,1200])
xline(mean(track_score),'r','LineWidth',3);
title(["Mean: "+num2str(mean(track_score))])
xlabel('Performance Reward')
ylabel('Frequency')
subplot(1,3,2)
histogram(track_score2,7)
xlim([-5,130])
ylim([0,1200])
xline(mean(track_score2),'r','LineWidth',3);
title(["Mean: "+num2str(mean(track_score2))])
xlabel('Performance Reward')
subplot(1,3,3)
histogram(track_score3,7)
xlim([-5,130])
ylim([0,1200])
xline(mean(track_score3),'r','LineWidth',3);
title(["Mean: "+num2str(mean(track_score3))])
xlabel('Performance Reward')
mean(track_score)
mean(track_score2)
mean(track_score3)
