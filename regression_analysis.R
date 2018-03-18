library(data.table)
library(bit64)
library(dplyr)
require(ggplot2)
library(xgboost)
library(Matrix)

#"jitter", "rx_bitrate", "rssi"

data_filename = "./data/data.csv"
in_data = fread(data_filename, header = T)


### Distribution of jitter
### Check for outlier values
ggplot(in_data, aes(jitter)) + geom_histogram()
#ggplot(data, aes(x = "jitter", y = jitter)) + geom_boxplot()


### Eye Balling Feature Type, Stats and Values
summary(in_data)
str(in_data)


### Clean Data 
### Remove Columns which doesn't have any variation or seems irrelevant
### and Rows with NA values
irrelevant_colnames = c("protocol", "bandwidth", "mcs_type", "band", 
                        "client_tx_airtime", "tput_theta1", "roam_count", 
                        "vlan_id", "num_locating_aps", "proto", "key_mgmt", "id")
string_columns = c("protocol", "mcs_type", "os", "family", "manufacture", "proto", "id")
flagged_colnames = c("epoch", "pq_dropped", "last_seen", "rt")

cleaned_data = in_data[,!colnames(in_data) %in% irrelevant_colnames, with = FALSE]
cleaned_data = cleaned_data[complete.cases(cleaned_data), ]

### Remove observations having values greater than 97.5 percentile
quantile_step = 0.025
quantile_threshold = "97.5%"
quant_probs = seq(0, 1, quantile_step)
jitter_percentile_values = quantile(cleaned_data$jitter, probs = quant_probs)
jitter_outlier_threshold = jitter_percentile_values[quantile_threshold]
n_important_features = 10

### Remove observations for outlier 
cleaned_data = cleaned_data[cleaned_data$jitter < jitter_outlier_threshold]
ggplot(cleaned_data, aes(jitter)) + geom_histogram()


### Correlation based Importance
cleaned_data_matrix = sparse.model.matrix( ~ . + -1, data = cleaned_data)
cor_jitter_other_columns = cor(x = as.matrix(cleaned_data_matrix))[, "jitter"]
ordered_coorelation_values = sort(abs(cor_jitter_other_columns), decreasing = T)
correlation_important_features = names(ordered_coorelation_values)[-1][1:n_important_features]


### Data preparation for XGBoost
train_data = cleaned_data %>% select(-jitter)
feature_matrix = sparse.model.matrix( ~ . + -1, data = train_data)
target = cleaned_data$jitter


### XG Boost Feature Importance
xgb.model = xgboost(data = feature_matrix, label = target, max.depth = 1, eta = 0.8, 
                    objective = "reg:linear", nrounds = 2500)

unscaled_importance_matrix <- xgb.importance(model = xgb.model)
unscaled_topn_features = colnames(feature_matrix)[as.integer(unscaled_importance_matrix[1:n_important_features]$Feature)]
xgb.plot.importance(importance_matrix = unscaled_importance_matrix, top_n = n_important_features)


### XG Boost Feature Importance with Scaling
scaled_feature_matrix = scale(feature_matrix)
scaled_xgb.model = xgboost(data = scaled_feature_matrix, label = target, max.depth = 1, eta = 0.8, 
                    objective = "reg:linear", nrounds = 2500)
scaled_importance_matrix <- xgb.importance(model = scaled_xgb.model)
scaled_topn_features = colnames(scaled_feature_matrix)[as.integer(scaled_importance_matrix[1:n_important_features]$Feature)]
xgb.plot.importance(importance_matrix = scaled_importance_matrix, top_n = n_important_features)

