%%Alpaslan Erdag

%% --------------- Importing the dataset -------------------------
% ---------------------------- Code ---------------------------
clc
clear all
data = readtable('dataarranged.xls')
% data= datarow(:,1:5);


%________________________________________________________________
%________________________________________________________________

%%---------------Data Preprocessing -----------------------------
% -------------- Handling Missing Values ------Deleting NaN values in data --------------
 % I realized that there were some NaN values, so first I got rid of them
 % below
%  Deleting NaN values in data --------------
% ---------------------------- Code ---------------------------
% 
 complete_data = rmmissing(data);
 complete_data = rmmissing(data,1);
 data = complete_data; 
 

%% -------------- Handling Outliers-------------------------------
% There are some outliers of course. With the code below, i replaced these
% outliers with mean values... There are some other methods withs this
% function. You can check..
% ---------------------------- Code -----------------------------
% 
 Temp = filloutliers(data.Temperature,'clip')
 data.Temperature = Temp;
 hum = filloutliers(data.Humidity,'clip')
 data.Humidity = hum;
%  light = filloutliers(data.Light,'nearest') there are something wrong
%  about this feature, i didn't use this for outlier detection or
%  classification
%  data.Light = light;
  co2 = filloutliers(data.CO2,'clip')
 data.CO2 = co2;


%% -------------- Feature Scalling -------------------------------

% -------------- Standardization ----------------------
% There are several feature scaling/ normalization formulas, i used this
% below... new(x)= real(x) - mean(x) / standard deviation(x)
% ---------------------------- Code -----------------------------

 stand_temp = (data.Temperature - mean(data.Temperature))/std(data.Temperature)
 data.Temperature = stand_temp;
 stand_light = (data.Light - mean(data.Light))/std(data.Light)
 data.Light = stand_light;
 stand_CO2 = (data.CO2 - mean(data.CO2))/std(data.CO2)
 data.CO2 = stand_CO2;
 stand_hum = (data.Humidity - mean(data.Humidity))/std(data.Humidity)
 data.Humidity = stand_hum;
 
% After preprocess i record new dataset as "preprocessed"
writetable(data,'C:\Users\alp\preprocessed_data.xls'); 

%% -------------- Building Classifier ----------------------------
% ---------------------------- Code ---------------------------

%classification_model = fitcsvm(data,'Occupancy~ Humidity+CO2','KernelFunction','linear','BoxConstraint',1);
regression_model = fitrsvm(data, 'Occupancy~ Humidity+CO2', 'KernelFunction','linear');

%% -------------- Test and Train sets ----------------------------
% ----------------------------Build-in function Code ---------------------------
% 
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);
cross_validated_model = crossval(classification_model,'cvpartition',cv); 



%% -------------- Making Predictions for Test sets ---------------
% ---------------------------- Code ---------------------------

Predictions = predict(cross_validated_model.Trained{1},data(test(cv),1:end-1));



%% -------------- Analyzing the predictions ---------------------
% ---------------------------- Code ---------------------------

Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions);




%% -------------- Visualizing training set results --------------
%  ---------------------------- Code ---------------------------
 
labels = unique(data.Occupancy);
classifier_name = 'SVM (Training Results)';
 % please mention your classifier name here

Hum_range = min(data.Humidity(training(cv)))-1:0.01:max(data.Humidity(training(cv)))+1;
co2_range = min(data.CO2(training(cv)))-1:0.01:max(data.CO2(training(cv)))+1;

[xx1, xx2] = meshgrid(Hum_range,co2_range);
XGrid = [xx1(:) xx2(:)];
 
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
 
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
 
hold on
 
training_data = data(training(cv),:);
Y = ismember(training_data.Occupancy,labels(1,1));
 
 
scatter(training_data.Humidity(Y),training_data.CO2(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Humidity(~Y),training_data.CO2(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
 
xlabel('CO2');
ylabel('Humidty');
 
title(classifier_name);
legend off, axis tight
 


%% -------------- Visualizing testing set results ----------------
% ---------------------------- Code ---------------------------
 
labels = unique(data.Occupancy);
classifier_name = 'SVM (Testing Results)';
Hum_range = min(data.Humidity(training(cv)))-1:0.01:max(data.Humidity(training(cv)))+1;
co2_range = min(data.CO2(training(cv)))-1:0.01:max(data.CO2(training(cv)))+1;

[xx1, xx2] = meshgrid(Hum_range,co2_range);
XGrid = [xx1(:) xx2(:)];

predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);

figure

gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');

hold on

testing_data =  data(test(cv),:);
Y = ismember(testing_data.Occupancy,labels(1,1));
 
scatter(testing_data.Humidity(Y),testing_data.CO2(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(testing_data.Humidity(~Y),testing_data.CO2(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');


 
xlabel('CO2');
ylabel('Humidty');

title(classifier_name);
legend off, axis tight

 
%________________________________________________________________




















































