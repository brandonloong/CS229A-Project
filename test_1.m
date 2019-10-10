clear
data = load('data_nonlife_0.txt');
data_train = load('data_nonlife_train.txt');

% Visualize histograms of cs_sentencemth
% y = data(:,end);
% y_uniq = unique(y);
% y_sort = sort(y);
% y_sort_yrs = y_sort/12;

% Visualize how cs_sentencemth varies with ch_numcar (numerical data)
figure
plot(data_train(:,end-1),data_train(:,end),'.')
xlabel('Number of Incarcerations (ch_numcar)','Interpreter','none')
ylabel('Sentencing Length (cs_sentencemth)','Interpreter','none')
xlim([0 40])

% Clean up original data
% data002_0 = load('data002_0.txt');


