set.seed(349)

# Fix column names in data frame
colnames(winequality_white)[colnames(winequality_white) == "fixed acidity"] <- "fixed_acidity"
colnames(winequality_white)[colnames(winequality_white) == "volatile acidity"] <- "volatile_acidity"
colnames(winequality_white)[colnames(winequality_white) == "citric acid"] <- "citric_acid"
colnames(winequality_white)[colnames(winequality_white) == "residual sugar"] <- "residual_sugar"
colnames(winequality_white)[colnames(winequality_white) == "free sulfur dioxide"] <- "free_sulfur_dioxide"
colnames(winequality_white)[colnames(winequality_white) == "total sulfur dioxide"] <- "total_sulfur_dioxide"
colnames(winequality_white)

# Create regression task from white wine quality data
wine <- as_task_regr(winequality_white, target = "quality")
# Check for missing values in each column
missing_cols <- colSums(is.na(winequality_white))

# Output columns with missing values
print(missing_cols[missing_cols > 0])
# Use random forest as learner as it had the best results in the algorithm selection exercise
lrn_ranger = lrn("regr.ranger", sample.fraction = to_tune(0.1, 1), num.trees = to_tune(1, 2000),
                 mtry.ratio = to_tune(0.0, 1),  min.node.size = to_tune(lower = 1, upper = 20))

# Scale data
graph =po("yeojohnson", lower = to_tune(0, 10), upper = to_tune(10, 30)) %>>% lrn_ranger
# Visualize Pipeline
graph$plot(horizontal = TRUE)


# Change graph to learner for tuning
graph_learner = as_learner(graph)

# Define tuning configuration 
tuned = auto_tuner(
  tuner = tnr("mbo"),
  learner = graph_learner,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rsq"),
  #search_space = lts_ranger,
  terminator = trm("run_time", secs = 300)
)

# Unoptimized learner and featureless
unop_lrn = lrn("regr.ranger")
featureless = lrn("regr.featureless")


# Nested resampling
design = benchmark_grid(wine, c(tuned, unop_lrn, featureless), rsmp("cv", folds = 3))
bmr = benchmark(design)
bmr$aggregate()[, .(task_id, learner_id, regr.mse)]
autoplot(bmr, measure = msr("regr.mse"))


# Check degree that each featrue is skewed from normal distribution as well as min and max value for each feature
fixed_acidity = winequality_white$fixed_acidity
min(fixed_acidity) # 3.8
max(fixed_acidity)# 14.2
faskew = skewness(fixed_acidity)
faskew # 0.64

volatile_acidity = winequality_white$volatile_acidity
min(volatile_acidity) # 0.08
max(volatile_acidity) # 1.1
vaskew = skewness(volatile_acidity)
vaskew # 1.57

ca = winequality_white$citric_acid
min(ca) # 0 
max(ca) # 1.66
caskew = skewness(ca)
caskew # 1.28

rs = winequality_white$residual_sugar
min(rs) # 0.6
max(rs) # 65.8
rs_skew = skewness(rs)
rs_skew # 1.07

cl = winequality_white$chlorides
min(cl) # 0.009
max(cl) #0.346
cl_skew = skewness(cl)
cl_skew # 5.02

fsd = winequality_white$free_sulfur_dioxide
min(fsd) # 2
max(fsd) # 289
fsd_skew = skewness(fsd)
fsd_skew # 1.4

tsd = winequality_white$total_sulfur_dioxide
min(tsd) # 9
max(tsd) # 449
tsd_skew = skewness(tsd)
tsd_skew # 0.39

d = winequality_white$density
min(d) # 0.98711
max(d) # 1.03898
skewness(d) # 0.9771742

ph = winequality_white$pH
min(ph) # 2.72
max(ph) # 3.82
skewness(ph) # 0.4575022

s = winequality_white$sulphates
min(s) # 0.22
max(s) # 1.08
skewness(s) # 0.9765952

a = winequality_white$alcohol
min(a) # 8 
max(a) # 14.2 
skewness(a) # 0.4870435


# Benchmarking pipline with different learners all hyperparameters were taken from https://github.com/COSC5557/pipeline-optimization-salarjarhan/blob/main/Pipeline%20Optimization.R
lrn_knn = lrn("regr.kknn", k = to_tune(lower = 1, upper = 20), distance = to_tune(lower = 1, upper = 20))
lrn_svm = lrn("regr.svm", type = "eps-regression", kernel = "radial", cost = to_tune(lower = 0.1, upper = 10), epsilon = to_tune(lower = 0.01, upper = 2), gamma = to_tune(lower = 0.01, upper = 2))
lrn_xgboost = lrn("regr.xgboost", nrounds = to_tune(lower = 10, upper = 300), max_depth =  to_tune(lower = 1, upper = 20))
learners = list(lrn_knn, lrn_svm, lrn_xgboost)

tuned_learners = lapply(learners, function(lrn) {
  graph = po("yeojohnson", lower = to_tune(0, 10), upper = to_tune(10, 30)) %>>% lrn
  graph_learner = as_learner(graph)
  auto_tuner(
    tuner = tnr("mbo"),
    learner = graph_learner,
    resampling = rsmp("cv", folds = 5),
    measure = msr("regr.mse"),
    #search_space = lts_ranger,
    terminator = trm("run_time", secs = 300)
  )
})

# Nested resampling
design = benchmark_grid(wine, c(tuned, tuned_learners, unop_lrn, featureless), rsmp("cv", folds = 3))
bmr = benchmark(design)
bmr$aggregate()[, .(task_id, learner_id, regr.mse)]
autoplot(bmr, measure = msr("regr.mse"))
