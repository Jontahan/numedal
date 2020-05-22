from util.plot_data import plot_data_from_file

plot_data_from_file('out/diversity_over_time_rein_300steps/data.txt', xlabel='Episodes', x_end=30000, ylabel='Diversity Score')