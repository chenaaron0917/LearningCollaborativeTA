classdef learning < handle
    properties
        env
        gamma
        human_reward_w
        human_reward
        agents_pi
        T_pi
    end
    
    methods
        % constructor
        function learner = learning(env, gamma, human_reward_w, agents_pi, T_pi)
            if nargin == 5
                learner.env = env;
                learner.gamma = gamma;
                learner.human_reward_w = human_reward_w;
                learner.human_reward = learner.human_reward_w(learner.env.getOpLevel(learner.env.op_states));
                learner.agents_pi = agents_pi;
                learner.T_pi = T_pi;
            end
        end
        
        function [human_pi, human_Q] = policyIteration(learner, human_pi)
            if nargin<2
                human_pi = randi(learner.env.n_tasks, learner.env.n_op_states, 1);
            end
            n_iter = 0;
            state_values = learner.human_reward;
            while 1
                old_human_pi = human_pi;
                V_pi = learner.getVpi(human_pi, state_values);
                human_Q = learner.getQfromV(V_pi);
                [~, human_pi] = max(human_Q, [], 2);
                if isequal(old_human_pi, human_pi)
%                 if sum(abs(old_human_pi-human_pi)) <=2
                    break
                else
                    n_iter = n_iter + 1;
                    if n_iter > 30
                        break
                    end
%                     disp(strcat("Iter: ",num2str(n_iter)))
%                     disp(strcat("Reward: ",num2str(learner.human_reward_w')))
                end
            end
        end
        
        % policy evaluation
        function V_pi = getVpi(learner, human_pi, state_values_pi, delta)
            if nargin==3
                delta = 0.01;
            end
            if nargin==2
                state_values_pi = nan;
                delta = 0.01;
            end
            if isnan(state_values_pi)
                state_values_pi = learner.human_reward;
            end
            
            while 1
                max_delta = 0;
                old_vs = state_values_pi;
                for s_i = 1:learner.env.n_op_states
                    assign = learner.agents_pi(s_i,:);
                    assign(end+1) = human_pi(s_i);
                    team_assign = learner.env.getTeamAssign(assign);
                    Tsa = squeeze(learner.T_pi(s_i, :, team_assign));
                    state_values_pi(s_i) = learner.human_reward(s_i)+learner.gamma* (Tsa*state_values_pi);
                end
                max_delta = max(max(abs(old_vs - state_values_pi)), max_delta);
                if max_delta < delta
                    break
                end
            end
            V_pi = state_values_pi;
        end
        
        % get Q values given V
        function Q = getQfromV(learner, state_values)
            Q = zeros(learner.env.n_op_states, learner.env.n_tasks);
            for s_i = 1: learner.env.n_op_states
                assign = learner.agents_pi(s_i,:);
                assigns = repmat(assign, learner.env.n_tasks,1);
                assigns = [assigns (1:learner.env.n_tasks)'];
                team_assign = learner.env.getTeamAssign(assigns);
                Ts = squeeze(learner.T_pi(s_i, :, team_assign)); % state x actions
                Q_s = learner.human_reward(s_i) + learner.gamma * Ts' * state_values; % actions x 1
                Q(s_i, :) = Q_s';
            end
        end
        
        function Q_pi = getQforPi(learner, human_pi)
            V_pi = learner.getVpi(human_pi);
            Q_pi = learner.getQfromV(V_pi);
        end
        
        function log_prior = getLogPrior(learner, r_max, type)
            if isequal(type,'uniform')
                log_prior = log(1/(r_max)^length(learner.human_reward_w));
            end
            
            if isequal(type,'beta')
                p = betapdf(linspace(0,1,r_max+2+1),0.5,0.5);
                p2 = p(2:end-1)/sum(p(2:end-1));
%                 op_level = learner.env.getOpLevel(learner.env.op_states);
%                 w = learner.human_reward_w(op_level);
                log_prior = sum(log(p2(learner.human_reward+1)));
            end
        end
    end
end