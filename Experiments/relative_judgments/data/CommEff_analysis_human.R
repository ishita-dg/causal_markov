library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)
library(BayesFactor)


setwd("~/GitHub/causal_markov/Experiments/relative_judgments/data")

expt_folder = "pilot2"

N_test = 20

df = data.frame()
N_part = 0
RTs = c()
for (fn in dir(path = expt_folder)){
  N_part = N_part + 1
  x = read.csv(paste(expt_folder, '/', fn, sep = ''))
  df0 = data.frame(resp = tail(x$response[is.finite(x$response)], N_test)/100.0,
                   stim = tail(x$stimulus[is.finite(x$response)], N_test),
                   left_query = tail(x$left_resp_stimulus[is.finite(x$response)], N_test),
                   right_query = tail(x$right_resp_stimulus[is.finite(x$response)], N_test),
                   cond = tail(rep(unique(x$condition)[2], each = sum(is.finite(x$response))), N_test),
                   part_num = tail(rep( N_part , each = sum(is.finite(x$response))), N_test),
                   rt = tail(as.numeric(as.character((x$rt[is.finite(x$response)]))), N_test)
  )
  df = rbind(df, df0)
}

df$stim = sapply(df$stim,function(x) as.integer((substr(x, 6, 8))))
df$left_query = sapply(df$left_query, function(x) as.integer((substr(x, 18, 19))))
df$right_query = sapply(df$right_query, function(x) as.integer((substr(x, 18, 19))))
df = subset(df, stim == 41)

df$query = (df$left_query + df$right_query)/2 - 0.5
df$norm_resp = (df$resp * df$left_query%%2) + (1.0 - df$resp) * (1 - df$left_query%%2)

val1 = c(3, 21)
val2 = c(5, 23)
val3 = c(15, 33)
val4 = c(9, 27)
val5 = c(17,35)

classify <- function(a){
  if (a %in% val1){
    return ('P(y1 = 1|y2 = 0)')
  } else if (a %in% val2){
    return ('P(y1 = 1|y2 = 1)')
  } else if (a %in% val3){
    return ('P(y1 = 1|y2 = 0, x = 1)')
  }else if (a %in% val4){
    return ('P(y1 = 1|x = 1)')
  }else if (a %in% val5){
    return ('P(y1 = 1|y2 = 1, x = 1)')
  }
  else {return ('discard')}
}

df$query_type = sapply(df$query, classify)
df0 = subset(df, query_type != 'discard')

labs = c('P(y1 = 1|y2 = 0)', 'P(y1 = 1|y2 = 1)', 
        'P(y1 = 1|y2 = 0, x = 1)', 'P(y1 = 1|x = 1)', 'P(y1 = 1|y2 = 1, x = 1)')


remove_outliers <- function(x){
  return(x[!x %in% boxplot.stats(x)$out])
}

# d_summ = ddply(df0, .(query_type, cond), summarize, mean = mean(remove_outliers(resp)), sd = sd(remove_outliers(resp)), N = length(remove_outliers(resp)))
d_summ = ddply(df0, .(query_type, cond), summarize, mean = mean(resp), sd = sd(resp), N = length(resp))
d_summ$CIs = qnorm(.975)*d_summ$sd/sqrt(d_summ$N)
d_true = data.frame(query_type = labs,
                    cond = c(rep('true', 5)),
                    mean = c(0.5, 0.5, 0.75, 0.6364, 0.57143),
                    sd = c(rep(0, 5)),
                    N = c(rep(1000, 5)),
                    CIs = c(rep(0, 5)))
d_summ = rbind(d_summ, d_true)
d_summ$query_type = factor(d_summ$query_type, levels = labs)
p <- ggplot(d_summ, aes(x=query_type, y=mean, fill=cond)) + 
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=mean-CIs, ymax=mean+CIs), width=.2,
                position=position_dodge(.9))+
  scale_fill_manual(values = c("#33cc00", "#CC0000", "#003399"))+
  ylim(-0.0, 0.8)

p

ggsave(file = paste("conditionals", expt_folder ,".png", sep = ''), p)

# 
# p <- ggplot(df0, aes(x=query_type, y=resp, color=cond)) + 
#   geom_jitter(height=0.0, width=0.08) + 
#   scale_fill_manual(values = c("#33cc00", "#CC0000", "#003399"))+
#   ylim(-0.0, 0.8)
#  
# p
# ttestBF(subset(df, cond == 'negcorr')$resp, subset(df, cond == 'poscorr')$resp)
# ttestBF(subset(df, cond == 'negcorr' & symt == 'Sym')$resp, subset(df, cond == 'poscorr' & symt == 'Sym')$resp)
# ttestBF(subset(df, cond == 'negcorr' & symt == 'Asym')$resp, subset(df, cond == 'poscorr' & symt == 'Asym')$resp)


