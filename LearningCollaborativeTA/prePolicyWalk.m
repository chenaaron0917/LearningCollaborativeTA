clc;clear;close all
FRAMEWORK_ITER = 2;
rng(FRAMEWORK_ITER)
load initTrain.mat
% Demos = [];
% for i = 1:FRAMEWORK_ITER
%     filename = char("humanDemo"+num2str(FRAMEWORK_ITER));
%     load(filename)
%     Demos = [Demos Demo];
% end
% Demo = Demos;
filename = char("humanDemo"+num2str(FRAMEWORK_ITER));
load(filename)
if FRAMEWORK_ITER>1
    filename = char("improvedAgent"+num2str(FRAMEWORK_ITER-1));
    load(filename)
end
% load initTrain.mat, load humanDemo.mat
%% Policy Walk Initiation and Pre-Process
%% environment class
env = OperationEnv(env_param);
env.reset;
env_param.agent_capa

%% aggregate demos
demo = [];
for d_i = 1:length(Demo)
    demo = [demo; Demo{d_i}];
end

%% process policies for atuonomous agents
agents_pi = zeros(env.n_op_states, env.n_agents - 1);
for s_i = 1:env.n_op_states
    for ag = 1:env.n_agents - 1
        Q = allagent_Qs{ag};
        [~, opt_assign] = max(Q(s_i,:));
        agents_pi(s_i, ag) = opt_assign;
    end
end

%% process Transition Prob for each team action
T_pi = zeros(env.n_op_states, env.n_op_states, env.n_actions);
for s_i = 1:env.n_op_states
    for assign_i = 1:env.n_actions
        assign = env.actions(assign_i, :)';
        env.demand_state = env.op_states(s_i,:);
        env.getCapaState(assign);
        TaskTransProb = env.getTaskTransProb;
        for next_s_i = 1:env.n_op_states
            s = env.op_states(next_s_i,:);
            TS = 1;
            for task_i = 1:env.n_tasks
                TS = TS * TaskTransProb(task_i,s(task_i));
            end
            T_pi(s_i, next_s_i, assign_i) = TS;
        end
    end
end
%%
if FRAMEWORK_ITER == 1
    filename = char("prePolicyWalk"+num2str(FRAMEWORK_ITER));
    save(filename, 'env', 'agents_pi', 'T_pi', 'demo')
else
    filename = char("prePolicyWalk"+num2str(FRAMEWORK_ITER));
    save(filename, 'env', 'agents_pi', 'T_pi', 'demo','human_reward_w')
end
% save prePolicyWalk.mat env agents_pi T_pi demo
