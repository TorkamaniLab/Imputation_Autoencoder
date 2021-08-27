
library(ggplot2)
library(stringr)
library(argparse)

args <- commandArgs(trailingOnly=TRUE)

print_help <- function(){

    cat("Usage:\n")
    cat("    Rscript plot_evaluation_results_per_sample.R <tsv_files> <options>\n")
    cat("Example:\n")
    cat("    Rscript plot_evaluation_results_per_sample.R ./evaluation_results/*_per_sample.tsv --threshold 0.90 --custom_files custom_file1.tsv custom_file2.tsv --custom_names custom_name1 custom_name2\n")
    cat("Positional arguments:\n")
    cat("    <tsv_files> Evaluation result file names(s), multiple files supported.\n")
    cat("Options (all optional):\n")
    cat("    --threshold [float]: Minimum correlation threshold (WGS vs imputed MAF correl) between -1 and 1, default=0.90\n")
    cat("    --custom_files [str, list]: list of custom evaluation results from other tools\n")
    cat("    --custom_names [str, list]: list of names for the custom evaluation results from other tools (i.e. minimac)\n")
    q()




}
if(length(args)==0){
    print_help()
}


parser <- ArgumentParser(description='Plot model evaluation results.')

parser$add_argument('tsv_files', metavar='N', type="character", nargs='+',help="evaluation result file names(s).")
parser$add_argument('--threshold', type="double", default=0.90, help="Minimum correlation threshold (WGS vs imputed MAF correl) [default %default]")


if("--custom_files" %in% args){
    parser$add_argument('--custom_files', metavar='custom_files', type="character", nargs='+', default=NULL, help="list of custom evaluation results from other tools")
    parser$add_argument('--custom_names', metavar='custom_files', type="character", nargs='+', default=NULL, help="list of names for the custom evaluation results from other tools (i.e. minimac)")
}


parsed_args <- parser$parse_args(args)

custom_names <- NULL
custom_files <- NULL
if("--custom_files" %in% args){
    custom_names <- parsed_args$custom_names
    custom_files <- parsed_args$custom_files
}

args <- parsed_args$tsv_files
threshold <- parsed_args$threshold


print(parsed_args)


#imputed_ids     WGS_ids F-score concordance_P0  r2      precision       recall  TP      TN      FP      FN      TP_ratio        TN_ratio        FP_ratio        FN_ratio        RMSE
require(plyr)
func <- function(xx){
    return(data.frame(MEAN_Fscore = mean(xx$F.score), MEAN_concordance = mean(xx$concordance_P0), MEAN_r2 = mean(xx$r2), file_name = unique(xx$file_name)))
}

library(gtools)

#if(length(args>1)){
#    args <- mixedsort(args)
#}


i=1
model_id<-str_match(args[1], ".imputed.\\s*(.*?)\\s*.vcf_per_sample.tsv")[,2]
print(paste0("Processing model with model id: ",model_id))

data_to_plot <- read.table(args[1], header=TRUE, sep='\t', stringsAsFactors=FALSE)
data_to_plot$file_name <- rep(basename(args[1]), nrow(data_to_plot))
data_to_plot$Model <- rep(model_id, nrow(data_to_plot))

mydata <- data_to_plot

if(length(args>1)){

    for(tsv_file in args[-1]){
       model_id<-str_match(tsv_file, ".imputed.\\s*(.*?)\\s*.vcf_per_sample.tsv")[,2]
       print(paste0("Processing model with model id: ",model_id))
       i=i+1
       mydata <- read.table(tsv_file, header=TRUE, sep='\t', stringsAsFactors=FALSE)
       mydata$file_name <- rep(basename(tsv_file), nrow(mydata))
       mydata$Model <- rep(model_id, nrow(mydata))

       data_to_plot <- rbind(data_to_plot,mydata)

    }
}


if(length(custom_files)>0){
    for(i in c(1:length(custom_files))){
        model_id <- custom_names[i]
        mydata <- read.table(custom_files[i], header=TRUE, sep='\t', stringsAsFactors=FALSE)
        mydata$file_name <- rep(basename(custom_files[i]), nrow(mydata))
        mydata$Model <- rep(model_id, nrow(mydata))
        data_to_plot <- rbind(data_to_plot,mydata)
    }


}


#data_to_plot <- subset(data_to_plot, WGS_MAF >= 0.005)

correls <- as.data.frame(ddply(data_to_plot, .(Model), func))

best <- subset(correls, MEAN_r2 > threshold)

best

#breaks <- c(0,0.005,0.05,0.1,0.2,0.3,0.4,0.5)
#bins <- c("[0-0.005)","[0.005-0.05)", "[0.005-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)","[0.4-0.5)")
#bin_tags <- cut(data_to_plot$WGS_MAF, 
#                  breaks=breaks, 
#                  include.lowest=TRUE, 
#                  right=FALSE, 
#                  labels=bins)

#data_to_plot$MAF_bin <- bin_tags

se <- function(data, na.rm=TRUE){
    return(sd(data, na.rm=na.rm)/sqrt(length(na.omit(data))))
}


data_to_plot$Model <- as.character(data_to_plot$Model)
library(dplyr)    #  loads %>%


data_to_plot <- data_to_plot %>% mutate_if(is.numeric, function(x) ifelse(is.infinite(x), 0, x)) %>% as.data.frame()
data_to_plot <- data_to_plot %>% filter_at(vars(Model), any_vars(. %in% c(best$Model)))    




res <- data_to_plot %>% group_by(Model) %>%
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

options(repr.plot.width = 10, repr.plot.height = 10)

sorted_labels <- unique(mixedsort(as.character(res$Model)))
res$Model <- factor(res$Model, levels = sorted_labels)


head(res)

p1 <- ggplot(res, aes(x=Model, y=Mean_r2)) +
    geom_bar(stat="identity", alpha=0.5) + ylim(0,1.05) +
    geom_errorbar(aes(ymin=Mean_r2-SD_r2, ymax=Mean_r2+SD_r2,  width=.2)) +
    theme_classic(base_size = 15) +
    coord_flip()

p2 <- ggplot(res, aes(x=Model, y=Mean_Fscore)) +
    geom_bar(stat="identity", alpha=0.5) + ylim(0,1.05) +
    geom_errorbar(aes(ymin=Mean_Fscore-SD_Fscore, ymax=Mean_Fscore+SD_Fscore,  width=.2)) +
    theme_classic(base_size = 15) +
    coord_flip()

p3 <- ggplot(res, aes(x=Model, y=Mean_concordance)) +
    geom_bar(stat="identity", alpha=0.5) + ylim(0,1.05) +
    geom_errorbar(aes(ymin=Mean_concordance-SD_concordance, ymax=Mean_concordance+SD_concordance,  width=.2)) +
    theme_classic(base_size = 15) +
    coord_flip()


label <- c( rep("R-squared", nrow(res)), rep("F-score", nrow(res)), rep("Concordance", nrow(res)) )
res.m <- c(res$Mean_r2, res$Mean_Fscore, res$Mean_concordance)
res.s <- c(res$SD_r2, res$SD_Fscore, res$SD_concordance)
res.i <- c(as.character(res$Model), as.character(res$Model), as.character(res$Model))

res.merged <- data.frame(Metric=factor(label), Mean=res.m, SD=res.s, Model=factor(res.i))


p4 <- ggplot(res.merged, aes(x=Metric, y=Mean, fill=Model)) +
    geom_bar(stat="identity", position = position_dodge(), color="black", alpha=0.8) + ylim(0,1.05) +
    geom_errorbar(aes(ymin=Mean-SD, ymax=Mean+SD),position=position_dodge(.9), width = .2) +
    theme_classic(base_size = 15) +
    coord_flip()
   
head(res.merged)

ggsave("p1_sample.pdf", plot = p1, scale = 1, width = 8, height = 8, dpi = 300)
ggsave("p2_sample.pdf", plot = p2, scale = 1, width = 8, height = 8, dpi = 300)
ggsave("p3_sample.pdf", plot = p3, scale = 1, width = 8, height = 8, dpi = 300)
ggsave("p4_sample.pdf", plot = p4, scale = 1, width = 8, height = 8, dpi = 300)

ggsave("p1_sample.png", plot = p1, scale = 1, width = 8, height = 8, dpi = 300)
ggsave("p2_sample.png", plot = p2, scale = 1, width = 8, height = 8, dpi = 300)
ggsave("p3_sample.png", plot = p3, scale = 1, width = 8, height = 8, dpi = 300)
ggsave("p4_sample.png", plot = p4, scale = 1, width = 8, height = 8, dpi = 300)
