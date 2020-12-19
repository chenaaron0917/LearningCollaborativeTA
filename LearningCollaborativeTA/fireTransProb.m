function T = fireTransProb(n_task_levels, n_capa_levels)
    % Fire Transition Probability T = (fire_s'|fire_s,capa)
    % T = (capa, fire_s, fire_s')
    T = zeros(n_capa_levels, n_task_levels, n_task_levels);
    for capa = 1:n_capa_levels
        T(capa,1,1)=1; % if fire demandlevel reaches 1, stay at 1
        % If capa >= fire demand, 0.7 prob to reduce
        % If capa < fire demand, 0.8 prob to increase
        for demand = 2:n_task_levels
            if capa >= demand
                T(capa,demand,demand) = 0.3;
                T(capa,demand,demand-1) = 0.7;
            else
                T(capa,demand,demand) = 0.2;
                if demand == n_task_levels
                    T(capa,demand,demand) = 1;
                else
                    T(capa,demand,demand+1) = 0.8;
                end
            end
        end
    end
end