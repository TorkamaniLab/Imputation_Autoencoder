
library(ggplot2)


args = commandArgs(trailingOnly=TRUE)

if(length(args)==0){
    print("Usage: Rscript plot_evaluation_results_per_variant.R <tsv_files>")
    print("Example: Rscript plot_evaluation_results_per_variant.R ./evaluation_results/*_per_variant.tsv")

    q()
}

require(plyr)
func <- function(xx){
    return(data.frame(COR = cor(xx$IMPUTED_MAF, xx$WGS_MAF), file_name = unique(xx$file_name)))
}

library(gtools)

if(length(args>1)){
    args <- mixedsort(args)
}


i=1
data_to_plot <- read.table(args[1], header=TRUE, sep='\t', stringsAsFactors=FALSE)
data_to_plot$file_name <- rep(basename(args[1]), nrow(data_to_plot))
data_to_plot$Model <- rep(i, nrow(data_to_plot))

mydata <- data_to_plot

if(length(args>1)){

    for(tsv_file in args[-1]){
       i=i+1
       mydata <- read.table(tsv_file, header=TRUE, sep='\t', stringsAsFactors=FALSE)
       mydata$file_name <- rep(basename(tsv_file), nrow(mydata))
       mydata$Model <- rep(i, nrow(mydata))

       data_to_plot <- rbind(data_to_plot,mydata)

    }
}

data_to_plot <- subset(data_to_plot, WGS_MAF >= 0.005)

correls <- as.data.frame(ddply(data_to_plot, .(Model), func))

best <- subset(correls, COR > 0.9)

best

breaks <- c(0,0.005,0.05,0.1,0.2,0.3,0.4,0.5)
bins <- c("[0-0.005)","[0.005-0.05)", "[0.005-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)","[0.4-0.5)")
bin_tags <- cut(data_to_plot$WGS_MAF, 
                  breaks=breaks, 
                  include.lowest=TRUE, 
                  right=FALSE, 
                  labels=bins)

data_to_plot$MAF_bin <- bin_tags

se <- function(data, na.rm=TRUE){
    return(sd(data, na.rm=na.rm)/sqrt(length(na.omit(data))))
}


data_to_plot$Model <- as.character(data_to_plot$Model)
library(dplyr)    #  loads %>%


data_to_plot <- data_to_plot %>% mutate_if(is.numeric, function(x) ifelse(is.infinite(x), 0, x)) %>% as.data.frame()
data_to_plot <- data_to_plot %>% filter_at(vars(Model), any_vars(. %in% c(best$Model)))    



res <- data_to_plot %>% group_by(Model,MAF_bin) %>%
        summarise(Mean_r2 = mean(r2,na.rm=TRUE), 
                  SD_r2 = sd(r2, na.rm=TRUE),
                  SE_r2 = se(r2, na.rm=TRUE),
                  Mean_Fscore = mean(F.score,na.rm=TRUE),
                  SD_Fscore = sd(F.score,na.rm=TRUE),
                  SE_Fscore = se(F.score,na.rm=TRUE),
                  Mean_concordance = mean(concordance_P0,na.rm=TRUE),
                  SD_concordance = sd(concordance_P0,na.rm=TRUE),
                  SE_concordance = se(concordance_P0,na.rm=TRUE)) %>%
        ungroup %>%
        as.data.frame()

options(repr.plot.width = 15, repr.plot.height = 10)

p1 <- ggplot(res, aes(x=MAF_bin, y=Mean_r2, group=Model, color=Model)) +
    geom_line() +
    geom_errorbar(aes(ymin=Mean_r2-SE_r2, ymax=Mean_r2+SE_r2,  width=.1)) +
    theme_classic(base_size = 15)

p2 <- ggplot(res, aes(x=MAF_bin, y=Mean_Fscore, group=Model, color=Model)) +
    geom_line() +
    geom_errorbar(aes(ymin=Mean_Fscore-SE_Fscore, ymax=Mean_Fscore+SE_Fscore,  width=.1)) +
    theme_classic(base_size = 15)

p3 <- ggplot(res, aes(x=MAF_bin, y=Mean_concordance, group=Model, color=Model)) +
    geom_line() +
    geom_errorbar(aes(ymin=Mean_concordance-SE_concordance, ymax=Mean_concordance+SE_concordance,  width=.1)) +
    theme_classic(base_size = 15)


ggsave("p1.pdf", plot = p1, scale = 1, width = 15, height = 10, dpi = 300)
ggsave("p2.pdf", plot = p2, scale = 1, width = 15, height = 10, dpi = 300)
ggsave("p3.pdf", plot = p3, scale = 1, width = 15, height = 10, dpi = 300)

ggsave("p1.png", plot = p1, scale = 1, width = 15, height = 10, dpi = 300)
ggsave("p2.png", plot = p2, scale = 1, width = 15, height = 10, dpi = 300)
ggsave("p3.png", plot = p3, scale = 1, width = 15, height = 10, dpi = 300)


p4 <- ggplot(data_to_plot,aes(x=WGS_MAF, y=IMPUTED_MAF,col=Model))+
    geom_point(alpha=0.5) +
    geom_abline(intercept = 0, slope = 1.0) +
    theme_classic(base_size = 15)


ggsave("p4.pdf", plot = p4, scale = 1, width = 15, height = 10, dpi = 300)
ggsave("p4.png", plot = p4, scale = 1, width = 15, height = 10, dpi = 300)


data_to_plot2 <- as.data.frame(rbind(cbind(data_to_plot$IMPUTED_MAF, data_to_plot$Model), cbind(mydata$WGS_MAF, rep("WGS", nrow(mydata)))), stringsAsFactors = FALSE)
names(data_to_plot2) <- c('MAF','Model')
data_to_plot2$MAF <- as.numeric(data_to_plot2$MAF)
p5 <- ggplot(data_to_plot2, aes(x=MAF, colour=Model)) +
    geom_density(aes(y = ..ndensity..)) +
    theme_classic(base_size = 15)


ggsave("p5.pdf", plot = p5, scale = 1, width = 15, height = 10, dpi = 300)
ggsave("p5.png", plot = p5, scale = 1, width = 15, height = 10, dpi = 300)
