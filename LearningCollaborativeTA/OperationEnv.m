classdef OperationEnv < handle
    properties
        n_agents
        agent_capa
        
        n_tasks
        n_task_levels
        demand_state
        op_states
        n_op_states
        op_state
        
        actions
        n_actions
        
        n_capa_levels
        capa_state
        
        fireT
        rescueT
        
        episode_step = 0;
        max_step = 30;
    end
    
    methods
        % constructor
        function env = OperationEnv(env_param)
            if nargin == 1
                env.n_agents = env_param.n_agents;
                env.agent_capa = env_param.agent_capa;
                env.n_tasks = env_param.n_tasks;
                env.n_task_levels = env_param.n_task_levels;
                env.n_capa_levels = env_param.n_capa_levels;
                env.op_states = env_param.op_states;
                env.n_op_states = env_param.n_op_states;
                env.actions = env_param.actions;
                env.n_actions = env_param.n_actions;
            	env.fireT = env_param.fireT;
                env.rescueT = env_param.rescueT;
            end
        end
        
        % reset environment
        function reset(env)
            env.episode_step = 0;
            env.demand_state = env.n_task_levels*ones(1,env.n_tasks);
            env.getOpState;
        end
        
        % step
        function  [reward, op_state, step, done] = step(env,assign)
            env.episode_step = env.episode_step + 1;
            env.getCapaState(assign);
            tasks_trans_prob = env.getTaskTransProb;
            % T = Pr(s'|s,a) tasks_trans_prob
            % evolve state, get s' and reward
            cumProb = cumsum(tasks_trans_prob,2);
            r = rand(env.n_tasks,1);
            [~,idx] = max(r<cumProb,[],2);
            env.demand_state = idx';
            reward = env.getReward;
            env.getOpState;
            op_state = env.op_state;
            step = env.episode_step;
            done = env.episode_step>=env.max_step;
        end
        
        % compute capabilities for each task given assignment
        function getCapaState(env, assign)
            A = (assign==1:env.n_tasks); % onehotencode
            % unassigned task has 0 capa, hence +1 and caped at 4
            capa_s = env.agent_capa*A+1; 
            env.capa_state = min(capa_s, env.n_capa_levels);
        end
        
        function tasks_trans_prob = getTaskTransProb(env)
            tasks_trans_prob = zeros(env.n_tasks,env.n_task_levels);
            for task = 1:env.n_tasks
                if mod(task,2)==1 % fire
                    capa_level = env.capa_state(1,task);
                    fire_demand_level = env.demand_state(task);
                    fT = env.fireT(capa_level,fire_demand_level,:);
                    tasks_trans_prob(task,:) = squeeze(fT);
                end
                if mod(task,2)==0 % rescue
                    capa_level = env.capa_state(2,task);
                    rescue_demand_level = env.demand_state(task);
                    fire_demand_level = env.demand_state(task-1);
                    rT = env.rescueT(capa_level,fire_demand_level,rescue_demand_level,:);
                    tasks_trans_prob(task,:) = squeeze(rT);
                end
            end
        end
        
        % reward on deman_state
        function reward = getReward(env)
            if isequal(env.demand_state,ones(1,env.n_tasks))
                reward = 10;
            else
                reward = 0;
            end
        end
        
        % convert demand state to operation state
        function getOpState(env)
            [~, env.op_state] = ismember(env.demand_state, env.op_states, 'rows');
        end
        
        % convert agents actions to team assignment
        function team_assign = getTeamAssign(env, agent_assign)
            [~, team_assign] = ismember(agent_assign, env.actions, 'rows');
        end
        
        % get operation state level based on task level
        function level = getOpLevel(env, state)
            level = sum(state,2) - size(state,2) + 1;
        end
    end
end