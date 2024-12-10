library(tidyverse)
library(vroom)
library(forecast)
library(patchwork)
library(embed)
library(tidymodels)
library(stringr)

data <- vroom("data.csv")

data$matchup <- ifelse(str_detect(data$matchup, 'vs.'), # create Home vs Away variable
                       'Home', 'Away') 
data$season <- substr(str_split_fixed(data$season, '-',2)[,2],2,2) # extract year number from season

# split training and testing data
train_data <- data[is.na(data$shot_made_flag) == FALSE, ]
test_data <- data[is.na(data$shot_made_flag) == TRUE, ]

train_data$shot_made_flag <- as.factor(train_data$shot_made_flag)

recipe <- recipe(shot_made_flag ~ ., data = train_data) %>%
  step_mutate(time_remaining = (minutes_remaining*60)+seconds_remaining, # combine time remaining into one variable
              shot_distance = sqrt((loc_x/10)^2 + (loc_y/10)^2), # distance from the hoop based on x and y location coords
              angle = case_when(loc_x == 0 ~ pi / 2, TRUE ~ atan(loc_y / loc_x)), # angle from the hoop
              game_number = as.numeric(game_date) # give each game a number identifier that increases proportional to time
              ) %>%
  step_mutate_at(c('game_id', 'playoffs', 'season', 'shot_type', 'shot_zone_area', 
                   'shot_zone_basic','matchup', 'opponent', 
                   'shot_id', 'period'), fn = factor) %>% # make factors out of these variables
  step_rm(c('shot_id', 'team_id', 'team_name', 'shot_zone_range', 'lon', 'lat',
            'seconds_remaining', 'minutes_remaining', 'game_event_id',
            'game_id', 'game_date','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y')) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) 


trees_model <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

trees_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(trees_model) 

# set up parallel computing
library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

tuning_grid <- grid_regular(
  mtry(range=c(1,(ncol(train_data)-1))),
  min_n(),
  levels = 3)

folds <- vfold_cv(train_data, v = 3, repeats=1)

CV_results <- trees_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_wf <- trees_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

predictions <- final_wf %>%
  predict(new_data = test_data, type = "prob")

submission <- predictions %>%
  bind_cols(., test_data) %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(x=submission, file="./KobeRFPreds.csv", delim=",")


