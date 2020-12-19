clc;clear;close all
FRAMEWORK_ITER = 1;
rng(FRAMEWORK_ITER)

if FRAMEWORK_ITER == 1
    load initTrain.mat
else
    load initTrain.mat
    filename = char("improvedAgent"+num2str(FRAMEWORK_ITER-1));
    load(filename)
end


%% Human Demonstration
% sim parameters
n_sims = 2;

% environment
env = OperationEnv(env_param);
env.reset;
env_param.agent_capa

%% Start collection
Demo = {}; % state, action histories
for sim_i = 1:n_sims
%     human_sa_pair = ones(env.max_step, 2);
    env.reset;
    op_state = env.op_state;
    eval_score = 0;
    done = 0;
    disp(strcat("Simulation #: ", num2str(sim_i)))
    while done ~= 1
        assign = zeros(env.n_agents,1);
        for ag = 1:env.n_agents-1
            Q = allagent_Qs{ag};
            [maxQ, opt_assign] = max(Q(op_state,:));
            assign(ag) = opt_assign;
        end
        
        % collect your decision
        disp(strcat("Operation step: ", num2str(env.episode_step)))
        disp(strcat("Operation state: ", num2str(env.demand_state)))
        human_assign = input("Pick a task 1, 2, 3, 4 to go: ");
        state_level  = getStateLevel(env.demand_state);
        assign(end) = human_assign;
        
        % evolve state
        [env_reward, next_op_state, step, done] = env.step(assign);
        
        % store human demo
        human_sa_pair(env.episode_step,:) = [op_state,human_assign];
        op_state = next_op_state;
        if op_state == 1
            human_sa_pair(end+1, :) = [op_state, 1];
            break
        end
    end
    Demo{sim_i} = human_sa_pair;
end
filename = char("humanDemo"+num2str(FRAMEWORK_ITER));
save(filename, 'Demo')
%% Function
% reward based on sum of the task demand levels
function level = getStateLevel(state)
    level = sum(state) - length(state) + 1;
end

