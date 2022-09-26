import pandas as pd
import numpy as np

from genomeai import read_tile

method_index = ['phased_minimac', 'phased_beagle', 'phased_impute']

tile = read_tile("tile.yaml")
summary = pd.read_csv(f"{tile.name}_summary.csv", index_col=0)
summary.set_index(summary.index.map(lambda x: f'model_{x}'), inplace=True)

# Append "plots_{i}" to summary dataframe.
for i in [1,2,3]:
    info = pd.read_csv(f"plots_{i}/overall_results_per_model.tsv", sep="\t",
                       index_col=0)
    # add suffix to each column name
    info.columns = info.columns.map(lambda x: x+f'_{i}')
    summary = summary.join(info, how='outer')

""" Note on current data values:
>>> summary.columns
Index(['number', 'value', 'params_L1', 'params_L2', 'params_activation',
       'params_batch_size', 'params_beta', 'params_decay_rate',
       'params_disable_alpha', 'params_flip_alpha', 'params_gamma',
       'params_inverse_alpha', 'params_learn_rate', 'params_loss_type',
       'params_n_layers', 'params_optimizer_type', 'params_rho',
       'params_size_ratio', 'state', 'Mean_r2_1', 'SD_r2_1', 'SE_r2_1',
       'Mean_Fscore_1', 'SD_Fscore_1', 'SE_Fscore_1', 'Mean_concordance_1',
       'SD_concordance_1', 'SE_concordance_1', 'Mean_r2_2', 'SD_r2_2',
       'SE_r2_2', 'Mean_Fscore_2', 'SD_Fscore_2', 'SE_Fscore_2',
       'Mean_concordance_2', 'SD_concordance_2', 'SE_concordance_2',
       'Mean_r2_3', 'SD_r2_3', 'SE_r2_3', 'Mean_Fscore_3', 'SD_Fscore_3',
       'SE_Fscore_3', 'Mean_concordance_3', 'SD_concordance_3',
       'SE_concordance_3'],
      dtype='object')
"""

def avg_cols(df):
    # compute the average over all columns with data values present
    s = df.fillna(0).sum(1)  # sum of observations
    n = (1-df.isna()).sum(1) # observations
    na = n == 0
    s /= (n+na) # safely divide
    s[na] = np.nan # set missing data back to nan
    return s

summary['avg_r2'] = avg_cols( summary[[f'Mean_r2_{i}' for i in [1,2,3]]] )
summary.sort_values('avg_r2', inplace=True, ascending=False)
summary.to_csv('summary.csv')

# drop non-models
#summary.drop(method_index, axis=0, inplace=True)
# pull only complete models
models = summary.loc[summary.state == 'COMPLETE']

#m_id = int(models.iloc[0].number)
best = models.iloc[0].name

with open('model_r2pop.txt', 'w') as f:
    for name, row in models.iterrows():
        f.write(f"{name} {row['avg_r2']}\n")

with open('competitor_r2pop.txt', 'w') as f:
    for s in method_index:
        f.write(f"{s} {summary['avg_r2'].loc[s]}\n")

print( f"Best model is {best}, new full training model name: {best}_F" )

#echo -e "Full training script created at ${train_script}.best"
#echo -e "Models mean r2 per feature across validation datasets listed at $train_script.model_r2pop.txt"
#echo -e "Competitors mean r2 per feature across validation datasets listed at $train_script.competitor_r2pop.txt"
#echo
#echo -e "To run the training:"
#echo -e "bash $file 1> $file.out 2> $file.log"
#echo -e "bash $file 1> $file.out 2> $file.log" > train_best.sh
#echo -e "Command stored at: $(readlink -e train_best.sh)"
