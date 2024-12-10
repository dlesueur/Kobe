library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(GGally)
library(ggplot2)
library(glmnet)
library(stacks)
library(recipes)
library(embed)
library(themis)

# load in data and clean
data <- vroom("data.csv")
train_data <- data[is.na(data$shot_made_flag) == FALSE, ]
test_data <- data[is.na(data$shot_made_flag) == TRUE, ]
train_data$shot_made_flag <- as.factor(train_data$shot_made_flag)


recipe <- recipe(shot_made_flag ~ ., data = train_data) %>%
  step_date(game_date, features = c("month", "year")) %>%
  step_mutate_at(c('action_type', 'combined_shot_type', 'game_event_id',
                   'game_id', 'playoffs', 'season', 'shot_type', 'shot_zone_area', 'shot_zone_basic',
                   'shot_zone_range', 'team_id', 'team_name', 'matchup', 'opponent', 'shot_id', 'game_date_month',
                   'game_date_year'), fn = factor) %>%
  step_rm(c('team_id', 'matchup', 'shot_id', 'game_event_id', 'team_name', 'game_date', 
            'lat', 'loc_x', 'loc_y', 'lon')) %>%
  step_other(all_nominal_predictors(), threshold = .005) %>%
  #step_lencode_mixed(all_nominal_predictors(), outcome=vars(shot_made_flag)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) #%>%
#step_smote(all_outcomes(), neighbors=4) #%>%
#step_pca(all_predictors(), threshold=.8)

# Logistic Regression
pen_model <- logistic_reg(
  mixture = tune(), 
  penalty = tune()
) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

# Decision Tree
tree_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# Random Forest
rf_model <- rand_forest(
  mtry = tune(),
  trees = 100,  # Fixed for speed
  min_n = tune()
) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

pen_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(pen_model)

# tree_workflow <- workflow() %>%
#   add_recipe(recipe) %>%
#   add_model(tree_model)

rf_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)

# Tuning Logistic Regression
pen_results <- pen_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_regular(penalty(), mixture(), levels = 5),
    metrics = metric_set(roc_auc)
  )

# # Tuning Decision Tree
# tree_results <- tree_workflow %>%
#   tune_grid(
#     resamples = folds,
#     grid = grid_regular(cost_complexity(), tree_depth(), min_n(), levels = 5),
#     metrics = metric_set(roc_auc)
#   )

# Tuning Random Forest
rf_results <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_regular(mtry(range = c(2, 10)), min_n(), levels = 5),
    metrics = metric_set(roc_auc)
  )

model_stack <- stacks() %>%
  add_candidates(pen_results) %>%  # Add logistic regression results
  #add_candidates(tree_results) %>% # Add decision tree results
  add_candidates(rf_results)       # Add random forest results


model_stack <- model_stack %>%
  blend_predictions(metric = metric_set(roc_auc))

model_stack <- model_stack %>%
  fit_members()

stacked_predictions <- model_stack %>%
  predict(new_data = test_data, type = "prob")

submission <- stacked_predictions %>%
  bind_cols(., test_data) %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(x = submission, file = "./KobeStackedPreds.csv", delim = ",")