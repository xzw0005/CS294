import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def plot_data(data, value="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    plt.legend(loc='best').draggable()
    plt.show()
    
def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, directory, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root, 'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            
            experiment_data.insert(len(experiment_data.columns), 'Unit', unit)
            experiment_data.insert(len(experiment_data.columns), 'Condition', condition or exp_name)
            
            datasets.append(experiment_data)
            unit += 1
    return datasets

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    args = parser.parse_args()
    
    use_legend = False 
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), "Must give a legend title for each set of experiments."
        use_legend = True
        
    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)
            
    if isinstance(args.value, list):
        values = args.value 
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value)
        
if __name__=="__main__":
    main()