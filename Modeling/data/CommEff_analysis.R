library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)


setwd("~/GitHub/causal_markov/Modeling/data")

fn = 'N_part47__expt_nameCommon_effect__NHID2__NONLINtanh__L20.0__test_epoch0__test_lr0.0__train_epoch500__train_lr0.02__train_blocks30__plot_data'

CE_data <- fromJSON(txt=fn)

binary = seq(0, 7)
in_paper = c(0, 3, 2, 6, 1, 5, 4, 7)
df = data.frame(P_ams = CE_data$P_ams,
                N_ams = CE_data$N_ams,
                hrms = CE_data$hrms,
                q = CE_data$q,
                stim = CE_data$stim)
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
# ggsave(file = paste("diff_preds", fn ,".png", sep = ''), p0)
p0

sym_bins = c(0,7)
asym_bins = c(4,5)

classify <- function(a){
  if (a %in% sym_bins){
    return ('Sym')
  } else if (a %in% asym_bins){
    return ('Asym')
  }
  else {return ('discard')}
}

N_part = as.integer( (substr(fn, 7, 8))) + 1
N_test = length(df$q) / N_part

df0 = data.frame(diffPs = c((df$norm_N_ams - df$norm_hrms),
                            (df$norm_P_ams- df$norm_hrms)),
                 condition = rep(c('Neg', 'Pos'), each = length(df$q)),
                 part_num = rep(c(seq(1, N_test)), each = 2*N_part),
                 symt = c(sapply(df$q_remap,classify), sapply(df$q_remap,classify))
)
df_sum0 <- ddply(df0, .(condition, symt, part_num), summarize, mean_pp = mean(diffPs)/0.125)
df_sum <- ddply(df_sum0, .(condition, symt), summarize, mean = mean(mean_pp), sd =  qnorm(.975)*sd(mean_pp)/sqrt(N_part))

p <- ggplot(df_sum, aes(x=condition, y=mean, fill=symt)) + 
  geom_bar(stat="identity", position=position_dodge()) + ylim(c(-0.5, 0.5))+
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2,
                position=position_dodge(.9))

# p

# ggsave(file = paste("diff_preds_summary", fn ,".png", sep = ''), p)


df0 = data.frame(diffPs = c((df$norm_N_ams),
                            (df$norm_P_ams)),
                 condition = rep(c('Neg', 'Pos'), each = length(df$q)),
                 part_num = rep(c(seq(1, N_test)), each = 2*N_part),
                 symt = c(sapply(df$q_remap,classify), sapply(df$q_remap,classify))
)
df0 = subset(df0, symt != 'discard')
df_sum0 <- ddply(df0, .(condition, symt, part_num), summarize, mean_pp = mean(diffPs))
df_sum <- ddply(df_sum0, .(condition, symt), summarize, mean = mean(mean_pp), sd =  qnorm(.975)*sd(mean_pp)/sqrt(N_part))

p <- ggplot(df_sum, aes(x=symt, y=mean, fill=condition)) + 
  geom_bar(stat="identity", position=position_dodge()) + ylim(c(-0.0, 0.2))+
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2,
                position=position_dodge(.9))

p

ggsave(file = paste("diff_preds_summary", fn ,".png", sep = ''), p)


# ********************************
# Predictions based on conditionals
# ********************************
# writing required conditionals in terms of joints
# Binary order is top cause Y, bottom cause Y, effct X 
dec2bin <- function(x) tail(as.integer(rev(intToBits(x))), 3)
decimal <- lapply(df$q_remap, dec2bin)

# Binary order is effect, bottom, top
decimal <- lapply(df$q, dec2bin)
df$y <- lapply(decimal, function(x) x[1])
df$z <- lapply(decimal, function(x) x[2])
df$y <- lapply(decimal, function(x) x[3])

find_dep_vars <- function(df, var){
  
  P <- function(x, z, y) {
    mean(var[df$x == x & df$z == z & df$y == y])
  }
  
  dep_vars = c()
  #P_Y1_g_Y0 
  dep_vars[1] = (
    (P(1, 0, 0) + P(1, 0, 1))/(P(1, 0, 0) + P(1, 0, 1) + P(0, 0, 0) + P(0, 0, 1)) + 
      (P(0, 1, 0) + P(0, 1, 1))/(P(0, 1, 0) + P(0, 1, 1) + P(0, 0, 0) + P(0, 0, 1))
    )/2
  #P_Y1_g_Y1 
  dep_vars[2] = (
    (P(1, 1, 0) + P(1, 1, 1))/(P(1, 1, 0) + P(1, 1, 1) + P(1, 0, 0) + P(1, 0, 1))+
      (P(1, 1, 0) + P(1, 1, 1))/(P(1, 1, 0) + P(1, 1, 1) + P(0, 1, 0) + P(0, 1, 1))
    )/2
  #P_Y1_g_Y0_X1 
  dep_vars[3] = (
    P(1, 0, 1)/(P(1, 0, 1) + P(0, 0, 1))+
      P(0, 1, 1)/(P(0, 1, 1) + P(0, 0, 1))
  )/2
  #P_Y1_g_X1 
  dep_vars[4] = (
    (P(1, 1, 1) + P(1, 0, 1))/(P(1, 1, 1) + P(1, 0, 1) + P(0, 1, 1) + P(0, 0, 1))+
      (P(1, 1, 1) + P(0, 1, 1))/(P(1, 1, 0) + P(0, 1, 1) + P(1, 0, 0) + P(0, 0, 1))
  )/2
  #P_Y1_g_Y1_X1 
  dep_vars[5] = (
    P(1, 1, 1)/(P(1, 1, 1) + P(0, 1, 1))+
      P(1, 1, 1)/(P(1, 1, 1) + P(1, 0, 1))
  )/2
  
  return(dep_vars)
}

true = find_dep_vars(subset(df, stim == 40), subset(df, stim == 40)$hrms)
poscorr = find_dep_vars(subset(df, stim == 40), subset(df, stim == 40)$P_ams)
negcorr = find_dep_vars(subset(df, stim == 40), subset(df, stim == 40)$N_ams)
lab = c('P(y1 = 1|y2 = 0)', 'P(y1 = 1|y2 = 1)', 
           'P(y1 = 1|y2 = 0, x = 1)', 'P(y1 = 1|x = 1)', 'P(y1 = 1|y2 = 1, x = 1)')

all_dep_vars = data.frame(preds = c(true, poscorr, negcorr),
                          condition = rep(c('True', 'Pos', 'Neg'), each = 5),
                          labels = factor(rep(lab, times = 3), levels = lab)
)

p <- ggplot(all_dep_vars, aes(x=labels, y=preds, fill=condition)) + 
  geom_bar(stat="identity", position=position_dodge()) + ylim(c(-0.1, 1.0))
p
# #P_Y1_g_Y0 
# dep_vars[1] = (sum(subset(df, y1 == 1 & y2 == 0)$P_ams) / sum(subset(df, y2 == 0)$P_ams) + sum(subset(df, y2 == 1 & y1 == 0)$P_ams) / sum(subset(df, y1 == 0)$P_ams))/2
# #P_Y1_g_Y1 
# dep_vars[2] = (sum(subset(df, y1 == 1 & y2 == 1)$P_ams) / sum(subset(df, y2 == 1)$P_ams) + sum(subset(df, y2 == 1 & y1 == 1)$P_ams) / sum(subset(df, y1 == 1)$P_ams))/2
# #P_Y1_g_Y0_X1 
# dep_vars[3] = (sum(subset(df, y1 == 1 & y2 == 0 & x == 1)$P_ams) / sum(subset(df, y2 == 0 & x == 1)$P_ams) + sum(subset(df, y2 == 1 & y1 == 0 & x == 1)$P_ams) / sum(subset(df, y1 == 0 & x == 1)$P_ams))/2
# #P_Y1_g_X1 
# dep_vars[4] = (sum(subset(df, y1 == 1 & x == 1)$P_ams) / sum(subset(df, x == 1)$P_ams) + sum(subset(df, y2 == 1 & x == 1)$P_ams) / sum(subset(df, x == 1)$P_ams))/2
# #P_Y1_g_Y1_X1 
# dep_vars[5] = (sum(subset(df, y1 == 1 & y2 == 1 & x == 1)$P_ams) / sum(subset(df, y2 == 1 & x == 1)$P_ams) + sum(subset(df, y2 == 1 & y1 == 1 & x == 1)$P_ams) / sum(subset(df, y1 == 1 & x == 1)$P_ams))/2

