

function [data,header] = import_data(file)
    path = 'C:\Users\a.l.schwamb\Box\CA\';

    header = pop_biosig(strcat(path,file),'importmex','on','importevent','off');
    data = header.data;
end