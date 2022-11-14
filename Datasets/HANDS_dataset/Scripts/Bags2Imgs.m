%%%%%% SCRIPT TO OPEN A ROSBAG, SELECT THE CORRECTED ROS TOPIC AND PERFORM
%%%%%% THE TEMPORAL ALIGNMENT OF THE TWO FRAMES

%% OPEN A ROSBAG

clear all
close all
clc

[file,path] = uigetfile('*.bag');
bagSelected1=rosbag(strcat(path,file));
% keep in mind that ROS topic are named after the corresponding kinect
% sensor, modality and finally data stream. If no changes are made to the
% iai_kinect2 launcher file, the correct ROS topic should be
% kinect2/qhd/image_color_rect and kinect2/qhd/image_depth_rect
KINECT = '/kinect2';

% this portion is to count the elements already saved in a directory, in
% order to give the new images an incremental number while saving.
a=dir([strcat(path,'saved_images') '\*.png']);
imnum=size(a,1)/2 + 1;

%% EXTRACTION OF TIMESTAMPS
% for each message in the topic, the code extract the timestamps and
% creates a couple of vectors of color timestamps and depth timestamps

bagColor=select(bagSelected1,'Topic',strcat(KINECT,'/qhd/image_color_rect'));
imagesColorMsg=readMessages(bagColor);
tcolor=[];
for i=1:size(imagesColorMsg,1)
    tcolor(i)=seconds(imagesColorMsg{i}.Header.Stamp);
end

bagDepth=select(bagSelected1,'Topic',strcat(KINECT,'/qhd/image_depth_rect'));
imagesDepthMsg=readMessages(bagDepth);
tdepth=[];
for i=1:size(imagesDepthMsg,1)
    tdepth(i)=seconds(imagesDepthMsg{i}.Header.Stamp);
end   

%% TEMPORAL ALIGNMENT

% chooses the minimum index number between the length of the color
% timestamps vector and the depth timestamps vector. This is necessary
% because the two topics may contain a different number of messages.
% M = minimum value
% I = corresponding index

[M,I] = min([length(tcolor), length(tdepth)]);
% creates an array which corresponds to the range 1 to M
min_IDX = 1 : M; 

% if I == 1 then the color vector is the shortest one
if I == 1
    % search the closer value with respect to the reference one (color).
    % returns the array of correspondences IDX and the array of temporal
    % distances DT.
    [IDX,DT] = knnsearch(tdepth',tcolor');
    % creates a matrix using the range min_IDX and the two new arrays
    IDXs_DT = [ min_IDX', IDX, DT ];
    % deletes the rows for which the temporal distance is > 66 ms
    IDXs_DT(IDXs_DT(:,3)>1, :) = [];
   
else
    % same as before but in this case the depth vector is the shortest one
    [IDX,DT] = knnsearch(tcolor',tdepth');
    IDXs_DT = [ IDX, min_IDX', DT ];
    IDXs_DT(IDXs_DT(:,3)>1, :) = [];
end

ii = 2;
% we have a reference stream k1 and a calibrating stream k2, for example if
% the shortest vector is the color one, the color frames are the reference
% stream and the depth stream is the calibrating one.
% if two frames of k1 refer to the same vale of k2, keep only the one with
% a minimum temporal distance and delete the other one.

while ii <= size(IDXs_DT,1) 
    if (IDXs_DT(ii,1) == IDXs_DT(ii-1,1) || IDXs_DT(ii,2) == IDXs_DT(ii-1,2))
        if (IDXs_DT(ii,3) > IDXs_DT(ii-1,3))
            IDXs_DT(ii,:) = [];
        else
            IDXs_DT(ii-1,:) = [];
        end
    else
        ii = ii+1;
    end
end

% deletes the non synchronized times for color and depth
color_filter = ones(1, length(tcolor));
color_filter(IDXs_DT(:,1)) = 0;
color_filter = logical(color_filter);
tcolor(color_filter)=[];

depth_filter = ones(1, length(tdepth));
depth_filter(IDXs_DT(:,2)) = 0;
depth_filter = logical(depth_filter);
tdepth(depth_filter)=[];

clear DT I IDX IDXs_DT II M min_IDX i ii ll 


%% IMAGES SAVING

% only saves synchronized frames for the color and the depth streams
name = imnum;
for i=1:size(imagesColorMsg,1)
    if color_filter(i)==0
        imagesColor(:,:,:,i)=readImage(imagesColorMsg{i});
        imwrite(imagesColor(:,:,:,i),strcat(path,persona,num2str(name,'%03d'),'_color.png'));
        name=name+1;
    end
end

name = imnum;
for i=1:size(imagesDepthMsg,1)
    if depth_filter(i)==0
        imagesDepth(:,:,i) = readImage(imagesDepthMsg{i});
        imwrite(imagesDepth(:,:,i),strcat(path,persona,num2str(name,'%03d'),'_depth.png'));
        name=name+1;
    end
end



