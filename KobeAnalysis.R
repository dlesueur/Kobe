library(tidymodels)
library(modeltime)
library(timetk)
library(readr)
library(embed)
library(patchwork)
library(vroom)
library(themis)
library(discrim)
library(recipes)

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

#prepped <- prep(recipe, training=train_data)


nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
            set_mode("classification") %>%
            set_engine("naivebayes") 


nb_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(nb_model)

tuning_params <- grid_regular(Laplace(),
                              smoothness(),
                              levels = 5)

folds <- vfold_cv(train_data, v = 6, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_params,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_nb_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

nb_predictions <- final_nb_wf %>%
  predict(new_data = test_data, type = "prob")







submission <- nb_predictions %>%
  bind_cols(., test_data) %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(x=submission, file="./KobeNaivePreds.csv", delim=",")
