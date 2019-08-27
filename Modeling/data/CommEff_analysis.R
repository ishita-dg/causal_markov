library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)


setwd("~/GitHub/causal_markov/Modeling/data")

fn = 'N_part9__expt_nameCommon_effect__NHID3__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch3__train_lr0.02__train_blocks200__plot_data'
CE_data <- fromJSON(txt=fn)

binary = seq(0, 7)
in_paper = c(0, 4, 2, 1, 6, 5, 3, 7)
df = as.data.frame(CE_data)
df$q_remap <- mapvalues(CE_data$q, from = binary,
                    to = in_paper)

plot <- ggplot(df, aes(y = hrms, x = factor(q_remap))) +
  # geom_point()+
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-0, 0.3))  + ggtitle("True probs") +
  theme(axis.text.x = element_blank())
plot


plot <- ggplot(df, aes(y = P_ams, x = factor(q_remap))) +
  # geom_point()+
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-0, 0.3))  + ggtitle("True probs") +
  theme(axis.text.x = element_blank())
plot


plot <- ggplot(df, aes(y = N_ams, x = factor(q_remap))) +
  # geom_point()+
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-0, 0.3))  + ggtitle("True probs") +
  theme(axis.text.x = element_blank())
plot