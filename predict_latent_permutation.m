%%% Code to predict the scores of domain-specific latent abilities
%%% from the properties of brain networks (clusters)

%% Import data
conn_all = importdata('SC.mat'); %import the structural/functional connection of the whole brain from all individuals (360 x 360 x number of individuals)
clusters = importdata('clusters.mat'); %import the clustering results (1x360 vector)

%% Calculate the structural/functional connection properties of brain networks (clusters)
for s = 1:length(conn_all)
    fprintf('\n Subject # %6.3f',s);
    conn_s = conn_all(:,:,s);
    conn_s = squeeze(conn_s);
    for i = 1:7
    edge1 = find(clusters==i);
        for j = 1:7
            clear conn_total_all
            edge2 = find(clusters==j);
            for y = 1:length(edge1)
                ind_y = edge1(y);
                for z = 1:length(edge2)
                    ind_z = edge2(z);
                    if ind_y == ind_z
                        conn_total = nan;
                    else
                        conn_total = conn_s(ind_y,ind_z);
                    end
                    conn_total_all(y,z) = conn_total;
                end
            end
            total_all{s,i,j} = conn_total_all; %the ourpur is conn_all_F for functional and conn_all_S for structural
        end
    end
end

%% Calculate the local properties of brain networks (clusters)
addpath('G:\My Drive\PhD\GCN\Script')
prop = importdata("properties.mat");
clusters = importdata('clusters_10_un_ori.mat');
for i = 1:7
    cl = find(clusters==i);
    prop_cl = properties(:,cl,:);
    prop_cl = reshape(prop_cl, 838, []);
    properties_cl{i,:,:} = prop_cl;
end

%% Prediction of domain-specific cognitive ability scores using functional properties of brain networks (clusters)
conn_all_F = importdata('conn_all_F.mat'); %import the connection properties
behav = importdata('latent.mat'); %import the ability scores
sub_all = importdata('sub_all.mat'); %import the shuffling index (the shuffle of training and testing samples)
for sh = 1:5 %index of the shuffles
    fprintf('\n Shuffle # %6.3f',sh);
    for conn_i = 1:7 %index of the brain networks (clusters)
        fprintf('\n Conn # %6.3f',conn_i);
        for conn_j = 1:7 %index of the brain networks (clusters)
            clear train_input
            clear test_input
            conn = conn_all_F(:,conn_i,conn_j);
            idx_tr = find(sub_all(:,sh)==1);
            idx_ts = find(sub_all(:,sh)==2);
            num_ts = length(idx_ts);
            train_beh = behav(idx_tr,:);
            test_beh = behav(idx_ts,:);
            train_inputw = conn(idx_tr);
            for tr_i = 1:length(idx_tr)
                s = train_inputw{tr_i,1};
                s = reshape(s,[],1);
                s(isnan(s)) = [];
                train_input(tr_i,:) = s;
            end
            test_inputw = conn(idx_ts);
            for ts_i = 1:length(idx_ts)
                s = test_inputw{ts_i,1};
                s = reshape(s,[],1);
                s(isnan(s)) = [];
                test_input(ts_i,:) = s;
            end
            parfor b = 1:5 %index of the abilities
                fprintf('\n Behavior # %6.3f',b);
                train_output = train_beh(:,b);
                test_output = test_beh(:,b);
                lambda = 0.01;
                Bw = lasso(train_input,train_output,'Lambda',lambda);
                pred_trainw = train_input*Bw(1:end);
                pred_testw = test_inputw*Bw(1:end);
                [Bw,FitInfow] = lasso(train_input,train_output,'CV',10);
                idxLambda1SEw = FitInfow.IndexMinMSE;
                coefw = Bw(:,idxLambda1SEw);
                coef0w = FitInfow.Intercept(idxLambda1SEw);
                pred_train = train_input*coefw + coef0w;
                for rn = 1:1000
                    idx_ts_per = randi(num_ts,1,num_ts);
                    test_in_per = test_input(idx_ts_per,:);
                    pred_test = test_in_per*coefw + coef0w;
                    [c_tr, p_tr] = corr(pred_train,train_output);
                    [c_ts, p_ts] = corr(pred_test,test_output);
                    c_tr_w(sh,conn_i,conn_j,b,rn) = c_tr;
                    c_ts_w(sh,conn_i,conn_j,b,rn) = c_ts;
                    p_tr_w(sh,conn_i,conn_j,b,rn) = p_tr;
                    p_ts_w(sh,conn_i,conn_j,b,rn) = p_ts;
               end
            end
        end
    end
end

%% Prediction of domain-specific cognitive ability scores using structural properties of brain networks (clusters)
conn_all_S = importdata('conn_all_S.mat'); %import the connection properties
prop = importdata('properties_cl.mat'); %import the local properties
behav = importdata('latent.mat'); %import the ability scores
sub_all = importdata('sub_all.mat'); %import the shuffling index (the shuffle of training and testing samples)
for sh = 1:5 %index of the shuffles
    fprintf('\n Shuffle # %6.3f',sh);
    for conn_i = 1:7 %index of the brain networks (clusters)
        fprintf('\n Conn # %6.3f',conn_i);
        for conn_j = 1:7 %index of the brain networks (clusters)
            clear train_input
            clear test_input
            clear train_input_c
            clear test_input_c
            Total output
            idx_tr = find(sub_all(:,sh)==1);
            idx_ts = find(sub_all(:,sh)==2);
            num_ts = length(idx_ts);
            train_beh = behav(idx_tr,:);
            test_beh = behav(idx_ts,:);
            Local properties
            prop_s = prop(conn_i);
            prop_t = prop(conn_j);
            prop_s = cell2mat(prop_s);
            prop_t = cell2mat(prop_t);
            prop_s_tr = prop_s(idx_tr,:);
            prop_s_ts = prop_s(idx_ts,:);
            prop_t_tr = prop_t(idx_tr,:);
            prop_t_ts = prop_t(idx_ts,:);
            Conention properties
            conn = conn_all_S(:,conn_i,conn_j);
            train_inputw = conn(idx_tr);
            for tr_i = 1:length(idx_tr)
                s = train_inputw{tr_i,1};
                s = reshape(s,[],1);
                s(isnan(s)) = [];
                train_input_c(tr_i,:) = s;
            end
            test_inputw = conn(idx_ts);
            for ts_i = 1:length(idx_ts)
                s = test_inputw{ts_i,1};
                s = reshape(s,[],1);
                s(isnan(s)) = [];
                test_input_c(ts_i,:) = s;
            end
            Total input
            if conn_i==conn_j
                train_input = [train_input_c prop_s_tr];
                test_input = [test_input_c prop_s_ts];
            else
                train_input = [train_input_c prop_s_tr prop_t_tr];
                test_input = [test_input_c prop_s_ts prop_t_ts];
            end
            Predictio model
            parfor b = 1:5 %index of the brain networks (clusters)
                fprintf('\n Behavior # %6.3f',b);
                train_output = train_beh(:,b);
                test_output = test_beh(:,b);
                lambda = 0.01;
                Bw = lasso(train_inputw,train_output,'Lambda',lambda);
                pred_trainw = train_inputw*Bw(1:end);
                pred_testw = test_inputw*Bw(1:end);
                [Bw,FitInfow] = lasso(train_input,train_output,'CV',10);
                idxLambda1SEw = FitInfow.IndexMinMSE;
                coefw = Bw(:,idxLambda1SEw);
                coef0w = FitInfow.Intercept(idxLambda1SEw);
                pred_train = train_input*coefw + coef0w;
                for rn = 1:1000
                    idx_ts_per = randi(num_ts,1,num_ts);
                    test_in_per = test_input(idx_ts_per,:);
                    pred_test = test_in_per*coefw + coef0w;
                    [c_tr, p_tr] = corr(pred_train,train_output);
                    [c_ts, p_ts] = corr(pred_test,test_output);
                    c_tr_w(sh,conn_i,conn_j,b,rn) = c_tr;
                    c_ts_w(sh,conn_i,conn_j,b,rn) = c_ts;
                    p_tr_w(sh,conn_i,conn_j,b,rn) = p_tr;
                    p_ts_w(sh,conn_i,conn_j,b,rn) = p_ts;
                end
            end
        end
    end
end