library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)


setwd("~/GitHub/causal_markov/Modeling/data")

fn = 'N_part23__expt_nameCommon_effect__NHID4__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch210__train_lr0.02__train_blocks200__plot_data'
CE_data <- fromJSON(txt=fn)

binary = seq(0, 7)
in_paper = c(0, 3, 2, 6, 1, 5, 4, 7)
df = as.data.frame(CE_data)
df$q_remap <- mapvalues(CE_data$q, from = binary,
                    to = in_paper)

# p <- ggplot(df, aes(y = hrms, x = factor(q_remap))) +
#   # geom_point()+
#   geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
#   ylim(c(-0, 0.3))  + ggtitle("True probs") +
#   theme(axis.text.x = element_blank())
# p

d_summary = ddply(df, ~q_remap, summarize, mean_P = mean(P_ams), mean_N = mean(N_ams), mean = mean(hrms))

df$norm_P_ams = df$P_ams/(sum(d_summary$mean_P))
df$norm_N_ams = df$N_ams/(sum(d_summary$mean_N))
df$norm_hrms = df$hrms/(sum(d_summary$mean))
                
p_P <- ggplot(df, aes(y = norm_P_ams, x = factor(q_remap))) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  geom_point(mapping = aes(x = factor(q_remap), y = norm_hrms)) +
  ylim(c(-0, 0.5))  + ggtitle("Exaggerate edge bias with pos corr AB") +
  theme(axis.text.x = element_blank())


p_N <- ggplot(df, aes(y = norm_N_ams, x = factor(q_remap))) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  geom_point(mapping = aes(x = factor(q_remap), y = norm_hrms)) +
  ylim(c(-0, 0.5))  + ggtitle("Reduce edge bias with neg corr AB") +
  theme(axis.text.x = element_blank())

p0 = grid.arrange(p_P, p_N, ncol = 1)
ggsave(file = "diff_preds.png", p0)
p0

