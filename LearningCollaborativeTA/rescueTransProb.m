function T = rescueTransProb(n_task_levels, n_capa_levels)
    % Rescue Transition Probability T = (rescue_s'|fire_s,rescue_s,capa)
    % T = (capa, fire_s, rescue_s, rescue_s')
    T = zeros(n_capa_levels, n_task_levels, n_task_levels, n_task_levels);
    for capa = 1:n_capa_levels
        for fire_demand = 1:n_task_levels
            % if fire demand is low <=2, can rescue
            if fire_demand<=ceil(n_task_levels/2)
                clear_rescue = zeros(1,n_task_levels);
                clear_rescue(1) = 1;
                % if rescue demand is 1, stay at 1
                T(capa,fire_demand,1,:) = clear_rescue;
                % reduce rescue demand based on rescue capa with prob 1
                for rescue_demand = 2:n_task_levels
                    if capa>=rescue_demand
                        T(capa,fire_demand,rescue_demand,:) = clear_rescue;
                    else
                        T(capa,fire_demand,rescue_demand,rescue_demand) = 1;
                    end
                end
            else
            % if fire demand is high 3 or 4, cannot rescue
                T(capa,fire_demand,:,:) = eye(n_task_levels);
            end  
        end
    end
end