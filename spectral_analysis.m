%% Clean up the environment
clearvars;
close all;
clc;

%% Import and Preprocessing
filename = 'CA (6)\JA4221NO.edf';
[data,header] = import_data(filename);

