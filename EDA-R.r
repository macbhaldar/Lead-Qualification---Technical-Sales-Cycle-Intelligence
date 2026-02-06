library(tidyverse)

df <- read.csv("data/sales_lead_rfp_dataset.csv")

glimpse(df)


df <- df %>%
  mutate(
    lead_score =
      solution_fit_score*40 +
      past_vendor_experience*20 +
      ifelse(technical_complexity <= 3, 15, 0) +
      ifelse(response_time_hours <= 24, 15, 0) +
      ifelse(estimated_deal_value_usd > 100000, 10, 0),
    lead_category = case_when(
      lead_score < 40 ~ "Low",
      lead_score < 70 ~ "Medium",
      TRUE ~ "High"
    )
  )

#Logistic Regression
model <- glm(
  rfp_win ~ technical_complexity + response_time_hours +
            solution_fit_score + estimated_deal_value_usd +
            past_vendor_experience,
  data = df,
  family = binomial
)

summary(model)

#Odds Ratios
exp(coef(model))


