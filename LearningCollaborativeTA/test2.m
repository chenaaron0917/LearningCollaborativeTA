clc;clear;close all
FRAMEWORK_ITER = 1;
Demos = [];
for i = 1:FRAMEWORK_ITER
    filename = char("humanDemo"+num2str(FRAMEWORK_ITER));
    load(filename)
    Demos = [Demos Demo];
end
Demo = Demos;