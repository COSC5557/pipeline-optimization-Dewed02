set.seed(349)

# Fix column names in data frame
colnames(winequality_white)[colnames(winequality_white) == "fixed acidity"] <- "fixed_acidity"
colnames(winequality_white)[colnames(winequality_white) == "volatile acidity"] <- "volatile_acidity"
colnames(winequality_white)[colnames(winequality_white) == "citric acid"] <- "citric_acid"
colnames(winequality_white)[colnames(winequality_white) == "residual sugar"] <- "residual_sugar"
colnames(winequality_white)[colnames(winequality_white) == "free sulfur dioxide"] <- "free_sulfur_dioxide"
colnames(winequality_white)[colnames(winequality_white) == "total sulfur dioxide"] <- "total_sulfur_dioxide"
names(winequality_white)

# Create regression task from white wine quality data
wine <- as_task_regr(winequality_white, target = "quality")


# Use random forest as learner as it had the best results in the algorithm selection exercise
lrn_ranger = lrn("regr.ranger", sample.fraction = to_tune(0.1, 1), num.trees = to_tune(1, 2000),
                 mtry.ratio = to_tune(0.0, 1),  min.node.size = to_tune(lower = 1, upper = 20))
# Hyperparameter search space for learner
lts_ranger = lts("regr.ranger.default")
lts_ranger 
# Scale data
graph = po("scale") %>>% lrn_ranger
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
  terminator = trm("run_time", secs = 1800)
)

# Unoptimized learner and featureless
unop_lrn = lrn("regr.ranger")
featureless = lrn("regr.featureless")


# Nested resampling
design = benchmark_grid(wine, c(tuned, unop_lrn, featureless), rsmp("cv", folds = 3))
bmr = benchmark(design)
bmr$aggregate()[, .(task_id, learner_id, regr.mse)]
autoplot(bmr, measure = msr("regr.mse"))
