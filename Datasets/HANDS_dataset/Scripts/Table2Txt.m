%%%%%% SCRIPT TO CONVERT A MATLAB LABELING SESSION IN A DATASET TABLE IN
%%%%%% TXT FORMAT
%% FROM LABELING SESSION TO TABLE

% loads the image paths
data = gTruth.DataSource.Source;
% loads the bounding boxes annotations
boxes = gTruth.LabelData;

% here loads a base empty table with the classes in the right order
base = load('base_table.mat');
base = base.labels;
base = [base; boxes];
% empties the first row which was the example one
base(1,:) = [];

depth = [];
% since the depth frames have the exact same path and name but have the
% "depth" suffix instead of the "color" one, here we add those paths
% stripping the color frame paths of their suffix and adding the new one
for i = 1:size(data,1)
    D = data(i);
    D = D{1};
    D = strcat(D(1:end-9),'depth.png');
    depth = [depth; {D}];
end

% here we finally create the correct MATLAB table of our dataset
% first column: rgb frames paths
% second column: depth frames paths
% from third to end: all the 29 classes. If the box exists for a given
% class, it is written as an array in correspondence of the class column.
% Otherwise there is an empty array
newtable = [data depth base];
newtable.Properties.VariableNames{1} = 'rgb';
newtable.Properties.VariableNames{2} = 'depth';
clear base boxes D data depth i labels

%% SHUFFLES DATASET

% performs the shuffle
p = randperm(size(newtable,1));
newtable = newtable(p,:);

%% SAVE THE CLASS HEADERS IN A SEPARATE FILE

headers = newtable.Properties.VariableNames;
headers = array2table(headers);
writetable(headers,'headers.txt','Delimiter',',');

%% SAVE THE TABLE CONTENTS IN A TXT FILE

% ATTENTION: this code writes the entire table in a .txt file.
% If you don't want the depth frames paths, just delete the column using:
% newtable(:,2) = [];

% TO CREATE TRAINING AND TEST DATASETS:
% just load the corresponding matlab tables (i. e. Subject1.mat etc),
% change their names and join them together using something like this:
% newtable = [subj1; subj2; ...]
% then reshuffle the new table!!!

riga = table2array(newtable);

for j = 1:size(riga,1)
    for i=1:size(riga,2)
        if isempty(riga{j,i})
            % if no box is present in that cell, writes the empty python list
            R{j,i} = '[0 0 0 0]';
        elseif i == 1
            R{j,i} = riga{j,i};
        else
            R{j,i} = mat2str(riga{j,i});
        end
    end
end

R = array2table(R);

% change the name of the output file here
writetable(R,'output.txt','Delimiter',',')
clear i j R riga

