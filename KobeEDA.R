library(tidymodels)
library(modeltime)
library(timetk)
library(readr)
library(embed)
library(patchwork)
library(vroom)

data <- vroom("data.csv")
train_data <- data[is.na(data$shot_made_flag) == FALSE, ]
test_data <- data[is.na(data$shot_made_flag) == TRUE, ]
train_data$shot_made_flag <- as.factor(train_data$shot_made_flag)

train_data %>%
  ggplot(mapping = aes(x=action_type)) +
  geom_bar()

train_data %>%
  ggplot(mapping = aes(x=combined_shot_type)) +
  geom_bar()

train_data %>%
  ggplot(mapping = aes(x=shot_zone_area)) +
  geom_bar()



dist <- sqrt((kobe$loc_x/10)^2 + (kobe$loc_y/10)^2)
kobe$shot_distance <- dist

#Creating angle column
loc_x_zero <- kobe$loc_x == 0
kobe['angle'] <- rep(0,nrow(kobe))
kobe$angle[!loc_x_zero] <- atan(kobe$loc_y[!loc_x_zero] / kobe$loc_x[!loc_x_zero])
kobe$angle[loc_x_zero] <- pi / 2