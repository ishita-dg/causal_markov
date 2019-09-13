library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)
library(BayesFactor)

setwd("~/GitHub/causal_markov/Experiments/data")

expt_folder = "Experiment1/"
N_test = 20

df = data.frame()
N_part = 0
RTs = c()
for (fn in dir(path = expt_folder)){
  N_part = N_part + 1
  x = read.csv(paste(expt_folder, fn, sep = ''))
  
  RT = sum(tail(as.numeric(x$rt[is.finite(x$response)]), N_test))
  
  RTs = append(RTs, RT)
  df0 = data.frame(resp = tail(x$response[is.finite(x$response)], N_test)/100.0,
                   stim = tail(x$stimulus[is.finite(x$response)], N_test),
                   query = tail(x$resp_stimulus[is.finite(x$response)], N_test),
                   cond = tail(rep(unique(x$condition)[2], each = sum(is.finite(x$response))), N_test),
                   part_num = tail(rep( N_part , each = sum(is.finite(x$response))), N_test),
                   rt = tail(as.numeric(x$rt[is.finite(x$response)]), N_test)
  )
  df = rbind(df, df0)
}


df$query = lapply(df$query, function(x) as.integer((substr(x, 6, 7))) - 1)
df$stim = lapply(df$stim, function(x) as.integer((substr(x, 6, 8))) - 1)

df_0 = df
sumRT = ddply(df, .(part_num), summarize, tRT = sum(rt))
y_parts = sumRT$part_num[sumRT$tRT > quantile(sumRT$tRT)[3]]
# df = subset(df, part_num %in% y_parts)
# df = subset(df, rt > quantile(rt)[2])
df = subset(df, stim == 40)

binary = seq(0, 7)
in_paper = c(0, 3, 2, 6, 1, 5, 4, 7)
df$q_remap <- mapvalues(unlist(df$query), from = binary,
                    to = in_paper)

d_summary0 = ddply(df, .(q_remap, cond, part_num), summarize, mean_pp = mean(resp))
d_summary = ddply(d_summary0, .(q_remap, cond), summarize, mean = mean(mean_pp), 
                   sd = qnorm(.975)*sd(mean_pp)/sqrt(N_part))
                

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

N_test = length(df$q) / N_part

df$symt = c(sapply(df$q_remap,classify))
df = subset(df, symt != 'discard')
df_sum0 <- ddply(df, .(cond, symt, part_num), summarize, mean_pp = mean(resp))
df_sum <- ddply(df_sum0, .(cond, symt), summarize, mean = mean(mean_pp), sd =  qnorm(.975)*sd(mean_pp)/sqrt(N_part))

p <- ggplot(df_sum, aes(x=symt, y=mean, fill=cond)) + 
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2,
                position=position_dodge(.9))

p

ggsave(file = paste("diff_preds_summary", fn ,".png", sep = ''), p)

ttestBF(subset(df, cond == 'negcorr')$resp, subset(df, cond == 'poscorr')$resp)
ttestBF(subset(df, cond == 'negcorr' & symt == 'Sym')$resp, subset(df, cond == 'poscorr' & symt == 'Sym')$resp)
ttestBF(subset(df, cond == 'negcorr' & symt == 'Asym')$resp, subset(df, cond == 'poscorr' & symt == 'Asym')$resp)


