library(jsonlite)
library(ggplot2)
require(gridExtra)
library(plyr)


setwd("~/GitHub/causal_markov/Experiments/relative_judgments/data")

expt_folder = "pilot1"
comments = c()
N_part = 0
for (fn in dir(path = expt_folder)){
  N_part = N_part + 1
  x = read.csv(paste(expt_folder, '/', fn, sep = ''))
  print(as.character(tail(x$responses, 1)))
}
