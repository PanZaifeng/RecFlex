# Data Synthesis

The script `data_generate.py` can generate the input dataset (i.e., the embedding lookup indices) of recommendation models with given `data_config.txt` and `table_config.txt`.

The `table_config.txt` file contains information about the shape of each embedding table within the model. Each line in the file specifies the number of rows and the embedding dimension of a table, separated by a comma.

The `data_config.txt` provides the lookup index distributions of the features. The format of each line is *feature_type,feature_num,configs*, where *feature_type* is one of the following:

* one-hot: for one-hot features, *configs* in each line only contain the coverage (i.e., proportion of features not absent) of the feature.
* multi-hot: for multi-hot features, the pooling factors of each feature follow a truncated normal distribution. The *configs* should specify the mean value, the standard deviation, and the maximum value of pooling factors, as well as the coverage of the feature.
* multi-hot-static: for these special multi-hot features, the pooling factors are fixed among the samples of a specific feature, if not absent. The *configs* only specify the fixed pooling factor and coverage of the feature.
* multi-hot-one-side: for these special multi-hot features, the pooling factors of each feature follow a one-sided truncated normal distribution and there is no upper bound for the pooling factors. The *configs* only specify the mean value and the standard deviation of pooling factors, as well as the coverage of the feature.

Besides, *feature_num* means the number of features that follow the same lookup index distribution of this line.

The bacth sizes of the generated datasets follow a truncated log-normal distribution. Users can pass arguments to control the batch size distribution.

We have provided all required configurations and the script under [`../examples/models`](../examples/models) to generate the datasets used in our paper.