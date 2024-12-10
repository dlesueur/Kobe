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
library(kknn)
library(discrim)
library(naivebayes)
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
  step_rm(c('team_id', 'matchup', 'shot_id', 'game_event_id', 'team_name', 'game_date')) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors=4) %>%
  step_pca(all_predictors(), threshold=.8)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
              set_mode("classification") %>%
              set_engine("naivebayes") 

nb_wf <- workflow() %>%
          add_recipe(recipe) %>%
          add_model(nb_model)

tuning_grid <- grid_regular(Laplace(), smoothness(),levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(roc_auc)) 

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_nb_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

nb_predictions <- final_nb_wf %>%
  predict(new_data = test_data, type = "class")


submission <- nb_predictions %>%
  bind_cols(., test_data) %>%
  select(shot_id, .pred_class) %>%
  rename(shot_made_flag = .pred_class)

vroom_write(x=submission, file="./KobePreds.csv", delim=",")
