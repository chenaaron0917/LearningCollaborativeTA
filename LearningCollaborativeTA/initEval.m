clc;clear;close all
rng(1);
load initTrain.mat
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
        assign = zeros(env.n_agents,1);
        for ag = 1:env.n_agents
            Q = allagent_Qs{ag};
            [maxQ, opt_assign] = max(Q(op_state,:));
            assign(ag) = opt_assign;
        end
        % evolve state
        [env_reward, next_op_state, step, done] = env.step(assign);

        % get reward from environment
        eval_score = eval_score + (gamma^step) * env_reward;

        op_state = next_op_state;
    end
    track_score(eval_i) = eval_score;
end

%%
hist(track_score)
