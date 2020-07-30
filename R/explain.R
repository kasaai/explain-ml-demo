library(tidyverse)
library(DALEX)
library(scales)
library(patchwork)

# Extract data and model files
untar("artifacts.tar.gz")

set.seed(2020)

testing_data <- readr::read_csv("artifacts/toy-model-testing-data.csv") %>% 
  sample_n(50000)

toy_model <- keras::load_model_tf("artifacts/toy-model/")

predictors <- c(
  "sex", "age_range", "vehicle_age", "make", 
  "vehicle_category", "region"
)

# Define custom prediction function for DALEX
custom_predict <- function(model, newdata) {
  predict(model, newdata, batch_size = 10000)
}

explainer_nn <- DALEX::explain(
  model = toy_model,
  data = testing_data,
  y = testing_data$loss_per_exposure,
  weights = testing_data$exposure,
  predict_function = custom_predict,
  label = "neural_net"
)

# Compute PDP
pdp_vehicle_age <- ingredients::partial_dependency(
  explainer_nn, 
  "vehicle_age",
  N = 10000,
  variable_splits = list(vehicle_age = seq(0, 35, by = 0.1))
)

# While the DALEX suite of package implement `plot()` methods for these objects
#   (e.g., try `plot(pdp_vehicle_age)` here), we can also manually create our own
#   plots.

pdp_plot <- as.data.frame(pdp_vehicle_age) %>% 
  ggplot(aes(x = `_x_`, y = `_yhat_`)) + 
  geom_line() +
  ylab("Average Predicted Loss Cost") +
  theme_bw() +
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank())

vehicle_age_histogram <- testing_data %>% 
  ggplot(aes(x = vehicle_age)) + 
  geom_histogram(alpha = 0.8) +
  theme_bw() +
  ylab("Count") +
  xlab("Vehicle Age")

# Piece together two plots using patchwork
pdp_plot <- pdp_plot / vehicle_age_histogram +
  plot_layout(heights = c(2, 1))

ggsave("output/pdp-plot.png", plot = pdp_plot)

fi <- ingredients::feature_importance(
  explainer_nn,
  loss_function = function(observed, predicted, weights) {
    sqrt(
      sum(((observed - predicted) ^ 2 * weights) /sum(weights))
    )
  },
  variables = predictors,
  B = 10,
  n_sample = 50000
)

# Again, here we create our own plot with custom aesthetics, but you can also
#   just `plot(fi)`
fi_plot <- fi %>% 
  as.data.frame() %>% 
  (function(df) {
    df <- df %>% 
      group_by(variable) %>% 
      summarize(dropout_loss = mean(dropout_loss))
    full_model_loss <- df %>% 
      filter(variable == "_full_model_") %>% 
      pull(dropout_loss)
    df %>% 
      filter(!variable %in% c("_full_model_", "_baseline_")) %>%
      ggplot(aes(x = reorder(variable, dropout_loss), y = dropout_loss)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      geom_hline(yintercept = full_model_loss, col = "red", linetype = "dashed")+
      scale_y_continuous(limits = c(full_model_loss, NA),
                         oob = rescale_none
      ) +
      xlab("Variable") +
      ylab("Dropout Loss (RMSE)") +
      coord_flip() +
      theme_bw() +
      NULL
  })

ggsave("output/fi-plot.png", plot = fi_plot)

# Grab a sample row of data to compute variable attributions
sample_row <- testing_data[1,] %>% 
  select(!!predictors)
breakdown <- iBreakDown::break_down(explainer_nn, sample_row)

# Likewise, you can `plot(breakdown)` here to see what the default plotting
#   behavior is
df <- breakdown %>% 
  as.data.frame() %>% 
  mutate(start = lag(cumulative, default = first(contribution)),
         label = formatC(contribution, digits = 2, format = "f")) %>% 
  mutate_at("label", 
            ~ ifelse(!variable %in% c("intercept", "prediction") & .x > 0,
                     paste0("+", .x),
                     .x)) %>% 
  mutate_at(c("variable", "variable_value"),
            ~ .x %>% 
              sub("Entre 18 e 25 anos", "18-25", .) %>% 
              sub("Passeio nacional", "Domestic passener", .) %>% 
              sub("Masculino", "Male", .))

breakdown_plot <- df %>% 
  ggplot(aes(reorder(variable, position), fill = sign,
             xmin = position - 0.40, 
             xmax = position + 0.40, 
             ymin = start, 
             ymax = cumulative)) +
  geom_rect(alpha = 0.4) +
  geom_errorbarh(data = df %>% filter(variable_value != ""),
                 mapping = aes(xmax = position - 1.40,
                               xmin = position + 0.40,
                               y = cumulative), height = 0,
                 linetype = "dotted",
                 color = "blue") +
  geom_rect(
    data = df %>% filter(variable %in% c("intercept", "prediction")),
    mapping = aes(xmin = position - 0.4,
                  xmax = position + 0.4,
                  ymin = start,
                  ymax = cumulative),
    color = "black") +
  scale_fill_manual(values = c("blue", "orange", NA)) +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") + 
  geom_text(
    aes(label = label, 
        y = pmax(df$cumulative,  df$cumulative - df$contribution)), 
    nudge_y = 10,
    hjust = "inward", 
    color = "black"
  ) +
  xlab("Variable") +
  ylab("Contribution") +
  theme(axis.text.y = element_text(size = 10))

ggsave("output/breakdown-plot.png", plot = breakdown_plot)
