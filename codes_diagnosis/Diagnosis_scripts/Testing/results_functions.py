# Get prediction metrics.

# region Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# endregion

# Source code credit for this function: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')

    return df_cm, fig

  
def export_test_classification_report_csv(test_classification_report, test_results_fullpath, trained_class_names, suffix = 'single'):

  classification_report_df_rows = []
  classification_report_df_columns = ['P', 'R', 'F', 'Samples']
  classification_report_stat_values = np.zeros((len(trained_class_names) + 3, 4))

  row_count = 0
  for key_class_report, value_class_report in test_classification_report.items():
    classification_report_df_rows.append(key_class_report)
    # print(key_class_report)
    # print(value_class_report)

    column_count = 0
    if not(type(value_class_report) == float):
      for test_metric, test_metric_value in value_class_report.items():
        classification_report_stat_values[row_count, column_count] = test_metric_value
        column_count = column_count + 1
    else:
        classification_report_stat_values[row_count, 2] = value_class_report

    row_count = row_count + 1

  classification_report_df = pd.DataFrame(data = classification_report_stat_values, index = classification_report_df_rows, columns = classification_report_df_columns)
  classification_report_df.to_csv(os.path.join(test_results_fullpath, 'classification_report_' + suffix + '.csv'))

