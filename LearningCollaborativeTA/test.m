clc;clear;close all

n_tasks = 4;
n_task_levels = 4;

n_states = n_tasks^n_task_levels;
States = sortrows(combinator(n_task_levels,n_tasks,'p','r'));

% for i = 1:n_states
%     a1 = mod(i, (n_tasks-1)^n_task_levels)
%     
% end