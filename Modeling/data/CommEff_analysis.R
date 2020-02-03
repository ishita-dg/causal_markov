library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)


setwd("~/GitHub/causal_markov/Modeling/data")

fn = "N_part47__expt_nameCommon_effect__NHID2__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch500__train_lr0.02__train_blocks30__plot_data"

CE_data <- fromJSON(txt=fn)
total_part = as.integer(substr(fn, 7, 8)) + 1

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
  geom_bar(stat="identity", position=position_dodge()) + ylim(c(-0.0, 0.4))+
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2,
                position=position_dodge(.9))

p

ggsave(file = paste("symAsym_preds", fn ,".png", sep = ''), p)


# ********************************
# Predictions based on conditionals
# ********************************
# writing required conditionals in terms of joints
# Binary order is top cause Y, bottom cause Y, effct X 
dec2bin <- function(x) tail(as.integer(rev(intToBits(x))), 3)
decimal <- lapply(df$q_remap, dec2bin)

# Binary order is effect, bottom, top
decimal <- lapply(df$q, dec2bin)
df$c <- lapply(decimal, function(x) x[1])
df$b <- lapply(decimal, function(x) x[2])
df$a <- lapply(decimal, function(x) x[3])

find_dep_vars <- function(df0, var){
  
  P <- function(a, b, c) {
    mean(var[df0$a == a & df0$b == b & df0$c == c], na.rm = TRUE)
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
    (P(1, 0, 1)/(P(1, 0, 1) + P(0, 0, 1)))+
      (P(0, 1, 1)/(P(0, 1, 1) + P(0, 0, 1)))
  )/2
  #P_Y1_g_X1 
  dep_vars[4] = (
    (P(1, 1, 1) + P(1, 0, 1))/(P(1, 1, 1) + P(1, 0, 1) + P(0, 1, 1) + P(0, 0, 1))+
      (P(1, 1, 1) + P(0, 1, 1))/(P(1, 1, 1) + P(0, 1, 1) + P(1, 0, 1) + P(0, 0, 1))
  )/2
  #P_Y1_g_Y1_X1 
  dep_vars[5] = (
    P(1, 1, 1)/(P(1, 1, 1) + P(0, 1, 1))+
      P(1, 1, 1)/(P(1, 1, 1) + P(1, 0, 1))
  )/2
  
  return(dep_vars)
}

df$part_num <- rep(seq(1, total_part), each = length(df$q)/total_part)
df_balanced <- subset(df, stim == 40)
helper_fdv <- function(var, part_nums){
  return(find_dep_vars(subset(df_balanced, part_num == part_nums[1]), var))
}

summ0 <- ddply(df_balanced, .(part_num), summarise, 
              true_vals = helper_fdv(hrms, part_num), 
              P_am_vals = helper_fdv(norm_P_ams, part_num),
              N_am_vals = helper_fdv(norm_N_ams, part_num))
lab = c('P(y1 = 1|y2 = 0)', 'P(y1 = 1|y2 = 1)', 
        'P(y1 = 1|y2 = 0, x = 1)', 'P(y1 = 1|x = 1)', 'P(y1 = 1|y2 = 1, x = 1)')
summ0$labels <- factor(rep(lab, times = total_part), levels = lab)

summ <- ddply(summ0, .(labels), summarise, 
              mean_true = mean(true_vals, na.rm = TRUE),
              mean_poscorr = mean(P_am_vals, na.rm = TRUE),
              mean_negcorr = mean(N_am_vals, na.rm = TRUE),
              sd_true = sd(true_vals, na.rm = TRUE),
              sd_poscorr = sd(P_am_vals, na.rm = TRUE),
              sd_negcorr = sd(N_am_vals, na.rm = TRUE))

all_dep_vars0 = data.frame(mean_pred = c(summ$mean_true, summ$mean_poscorr, summ$mean_negcorr),
                          sd_pred = c(summ$sd_true, summ$sd_poscorr, summ$sd_negcorr),
                          condition = rep(c('True', 'Pos', 'Neg'), each = 5),
                          labels = factor(rep(lab, times = 3), levels = lab)
)

# all_dep_vars$condition_color <- mapvalues(all_dep_vars$condition, from = c('True', 'Pos', 'Neg'),
                        # to = c("blue", "red", "green"))

all_dep_vars = subset(all_dep_vars0, condition != 'Neg')
p <- ggplot(all_dep_vars0, aes(x=labels, y=mean_pred, fill=condition)) + 
  geom_bar(stat="identity", position=position_dodge()) + 
  ylim(c(0.0, 0.8))+
  geom_errorbar(aes(ymin=mean_pred-1.96*sd_pred/sqrt(total_part), ymax=mean_pred+1.96*sd_pred/sqrt(total_part)), width=.2,
                position=position_dodge(.9))+
  #scale_fill_manual(values = c("#CC0000", "#003399"))
  scale_fill_manual(values = c("#33cc00", "#CC0000", "#003399"))


p
ggsave(file = paste("conditional_preds", fn ,".png", sep = ''), p)


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

