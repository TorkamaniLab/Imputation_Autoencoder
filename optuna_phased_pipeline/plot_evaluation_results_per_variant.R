
library(ggplot2)
library(stringr)
library(argparse)

args <- commandArgs(trailingOnly=TRUE)

print_help <- function(){

    cat("Usage:\n")
    cat("    Rscript plot_evaluation_results_per_variant.R <tsv_files> <options>\n")
    cat("Example:\n")
    cat("    Rscript plot_evaluation_results_per_variant.R ./evaluation_results/*_per_variant.tsv --threshold 0.90 --custom_files custom_file1.tsv custom_file2.tsv --custom_names custom_name1 custom_name2\n")
    cat("Positional arguments:\n")
    cat("    <tsv_files> Evaluation result file names(s), multiple files supported.\n")
    cat("Options (all optional):\n")
    cat("    --threshold [float]: Minimum correlation threshold (WGS vs imputed MAF correl) between -1 and 1, default=0.90\n")
    cat("    --custom_files [str, list]: list of custom evaluation results from other tools\n")
    cat("    --custom_names [str, list]: list of names for the custom evaluation results from other tools (i.e. minimac)\n")
    cat("    --custom_title [str]: main title for the plots\n")
    cat("    --out_dir [str]: output directory where to save de plots (default is .)\n")
    q()




}
if(length(args)==0){
    print_help()
}


parser <- ArgumentParser(description='Plot model evaluation results.')

parser$add_argument('tsv_files', metavar='N', type="character", nargs='+',help="evaluation result file names(s).")
parser$add_argument('--threshold', type="double", default=0.90, help="Minimum correlation threshold (WGS vs imputed MAF correl) [default %default]")
parser$add_argument('--custom_title', metavar='custom_title', type="character", default='', help="main title for the plots")
parser$add_argument('--out_dir', metavar='out_dir', type="character", default='.', help="output directory where to save de plots (default is .)")


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

if(dir.exists(file.path(parsed_args$out_dir)) == FALSE){
    print(paste0("creating directory ", parsed_args$out_dir))
    dir.create(parsed_args$out_dir)
}


require(plyr)
func <- function(xx){
    return(data.frame(COR = cor(xx$IMPUTED_MAF, xx$WGS_MAF), file_name = unique(xx$file_name)))
}

library(gtools)

#if(length(args>1)){
#    args <- mixedsort(args)
#}


i=1
#model_id<-str_match(args[1], ".imputed.\\s*(.*?)\\s*.vcf_per_variant.tsv")[,2]
# e.g. evaluation_output_1/model_9.vcf_per_variant.tsv
model_id<-str_match(args[1], "[ ./](model_[0-9]+)[ ./]")[,2]
print(paste0("Processing model with model id: ",model_id))

data_to_plot <- read.table(args[1], header=TRUE, sep='\t', stringsAsFactors=FALSE)
data_to_plot$file_name <- rep(basename(args[1]), nrow(data_to_plot))
data_to_plot$Model <- rep(model_id, nrow(data_to_plot))

mydata <- data_to_plot

if(length(args>1)){

    for(tsv_file in args[-1]){
       model_id<-str_match(tsv_file, "[ ./](model_[0-9]+)[ ./]")[,2]
       print(paste0("Processing model with model id: ",model_id))
       i=i+1
       mydata <- read.table(tsv_file, header=TRUE, sep='\t', stringsAsFactors=FALSE)
       mydata$REF_MAF <- NULL
       mydata$file_name <- rep(basename(tsv_file), nrow(mydata))
       mydata$Model <- rep(model_id, nrow(mydata))

       data_to_plot <- rbind(data_to_plot,mydata)

    }
}


if(length(custom_files)>0){
    for(i in c(1:length(custom_files))){
        model_id <- custom_names[i]
        mydata <- read.table(custom_files[i], header=TRUE, sep='\t', stringsAsFactors=FALSE)
        mydata$REF_MAF <- NULL
        mydata$file_name <- rep(basename(custom_files[i]), nrow(mydata))
        mydata$Model <- rep(model_id, nrow(mydata))
        data_to_plot <- rbind(data_to_plot,mydata)
    }


}


data_to_plot <- subset(data_to_plot, WGS_MAF >= 0.001)

correls <- as.data.frame(ddply(data_to_plot, .(Model), func))

best <- subset(correls, COR > threshold)

best

breaks <- c(0,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5)
#breaks <- c(0,0.001,0.005,0.01,0.05,0.1,0.5)

bins <- c("(0-0.001)", "[0.001-0.005)","[0.005-0.01)", "[0.01-0.05)", "[0.05-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)","[0.4-0.5)")
#bins <- c("(0-0.001)", "[0.001-0.005)","[0.005-0.01)", "[0.01-0.05)", "[0.05-0.1)", "[0.1-0.5)")

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

overall_res <- data_to_plot %>% group_by(Model) %>%
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


sorted_labels <- unique(mixedsort(as.character(res$Model)))
res$Model <- factor(res$Model, levels = sorted_labels)
overall_res$Model <- factor(overall_res$Model, levels = sorted_labels)


p1 <- ggplot(res, aes(x=MAF_bin, y=Mean_r2, group=Model, color=Model)) +
    geom_line() +
    geom_errorbar(aes(ymin=Mean_r2-SE_r2, ymax=Mean_r2+SE_r2,  width=.1)) +
    theme_classic(base_size = 15) + 
    labs(subtitle = parsed_args$custom_title, color = "") +
    guides(col = guide_legend(nrow = 40))

p2 <- ggplot(res, aes(x=MAF_bin, y=Mean_Fscore, group=Model, color=Model)) +
    geom_line() +
    geom_errorbar(aes(ymin=Mean_Fscore-SE_Fscore, ymax=Mean_Fscore+SE_Fscore,  width=.1)) +
    theme_classic(base_size = 15) +
    labs(subtitle = parsed_args$custom_title, color = "") +
    guides(col = guide_legend(nrow = 40))

p3 <- ggplot(res, aes(x=MAF_bin, y=Mean_concordance, group=Model, color=Model)) +
    geom_line() +
    geom_errorbar(aes(ymin=Mean_concordance-SE_concordance, ymax=Mean_concordance+SE_concordance,  width=.1)) +
    theme_classic(base_size = 15) +
    labs(subtitle = parsed_args$custom_title, color = "") +
    guides(col = guide_legend(nrow = 40))

p4 <- ggplot(data_to_plot,aes(x=WGS_MAF, y=IMPUTED_MAF,col=Model))+
    geom_point(alpha=0.5) +
    geom_abline(intercept = 0, slope = 1.0) +
    theme_classic(base_size = 15) +
    labs(subtitle = parsed_args$custom_title, color = "") +
    guides(col = guide_legend(nrow = 40))

data_to_plot2 <- as.data.frame(rbind(cbind(data_to_plot$IMPUTED_MAF, data_to_plot$Model), cbind(mydata$WGS_MAF, rep("WGS", nrow(mydata)))), stringsAsFactors = FALSE)
names(data_to_plot2) <- c('MAF','Model')
data_to_plot2$MAF <- as.numeric(data_to_plot2$MAF)

p5 <- ggplot(data_to_plot2, aes(x=MAF, colour=Model)) +
    geom_density(aes(y = ..ndensity..)) +
    theme_classic(base_size = 15) +
    labs(subtitle = parsed_args$custom_title, color = "") +
    guides(col = guide_legend(nrow = 40))


file_name <- file.path(parsed_args$out_dir,"results_per_MAF_bin.tsv")
print(paste0("Saving summary results per MAF bin in ", file_name))
write.table(res, file=file_name, quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t")

file_name <- file.path(parsed_args$out_dir, "overall_results_per_model.tsv")
print(paste0("Saving overall summary results per model in ", file_name))
write.table(overall_res, file=file_name, quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t")

file_name <- file.path(parsed_args$out_dir,"MAF_correls.tsv")
print(paste0("Saving MAF correl results in ", file_name))
write.table(correls, file=file_name, quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t")



plots <- list(p1,p2,p3,p4,p5)

for(i in 1:5){
    #doesn't work on summit because png needs X11
    #for(suffix in c(".pdf", ".png")){
    for(suffix in c(".pdf")){
        plot_name <- paste0("p",i,suffix)
        file_name <- file.path(parsed_args$out_dir, plot_name)
        print(paste0("Saving plot ",i, " in ", file_name))
        ggsave(file_name, plot = plots[[i]], scale = 1, width = 15, height = 10, dpi = 300)
    }
}

