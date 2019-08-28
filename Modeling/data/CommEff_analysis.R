library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)


setwd("~/GitHub/causal_markov/Modeling/data")

fn = 'N_part17__expt_nameCommon_effect__NHID4__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch210__train_lr0.02__train_blocks200__plot_data'

CE_data <- fromJSON(txt=fn)

binary = seq(0, 7)
in_paper = c(0, 3, 2, 6, 1, 5, 4, 7)
df = data.frame(P_ams = CE_data$P_ams,
                N_ams = CE_data$N_ams,
                hrms = CE_data$hrms,
                q = CE_data$q)
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
ggsave(file = paste("diff_preds", fn ,".png", sep = ''), p0)
p0

sym_bins = c(0, 1, 6, 7)

classify <- function(a){
  if (a %in% sym_bins){
    return ('Sym')
  } else{
    return ('Asym')
  }
}

df0 = data.frame(diffPs = c(df$norm_N_ams - df$norm_hrms, df$norm_P_ams- df$norm_hrms),
                 condition = rep(c('Neg', 'Pos'), each = length(df$q)),
                 symt = c(sapply(df$q_remap,classify), sapply(df$q_remap,classify))
)
df_sum <- ddply(df0, .(condition, symt), summarize, mean = mean(diffPs), sd = sd(diffPs))

p <- ggplot(df_sum, aes(x=condition, y=mean, fill=symt)) + 
  geom_bar(stat="identity", position=position_dodge()) 
  + ylim(c(-0.05, 0.05))
  # geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2,
  #               position=position_dodge(.9))

p + scale_fill_brewer(palette="Paired") + theme_minimal()

ggsave(file = paste("diff_preds_summary", fn ,".png", sep = ''), p)
