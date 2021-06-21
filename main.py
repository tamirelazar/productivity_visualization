import pandas as pd
import numpy as np
import plotly
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

if __name__ == '__main__':
    """ Read raw csv """
    data = pd.read_csv("dataset.csv", usecols=['Hour', 'Top-Level Project', 'Duration'],
                       parse_dates=['Hour'])
    data.Duration = pd.to_timedelta(data.Duration)

    """ create processed dataframe """
    processed_columns = data['Top-Level Project'].unique()
    processed_columns = np.append(processed_columns, 'Long')
    # print(processed_columns)
    processed_data = pd.DataFrame(data=0, columns=processed_columns, index=data['Hour'].unique())
    # print(processed_data)

    """ load aggregated durations"""
    for index, row in data.iterrows():
        project = row['Top-Level Project']
        duration = row['Duration']
        hour = row['Hour']

        if duration > pd.Timedelta('1 hour'):
            processed_data.at[hour, 'Long'] += int(duration.components.hours)
            processed_data.at[hour, project] += 60
            # print(pd.DatetimeIndex(duration).minute)
            # continue
        else:
            processed_data.at[hour, project] += int(duration.components.minutes)

        # a way to normalize long activities a little
        # if processed_data.at[hour, project] > 60:
        #     processed_data.at[hour, project] = 60
        #     processed_data.at[hour, 'Long'] += 1

        #
        # while duration > pd.Timedelta('0'):
        #     if duration > pd.Timedelta('1 hour'):
        #         curr_dur = 60
        #     else:
        #         curr_dur = duration.components.minutes
        #     if hour not in processed_data.index:
        #         new_row = pd.Series(processed_data.loc[(hour - pd.Timedelta('1 hour'))])
        #         new_row.name = hour
        #         print(new_row)
        #         processed_data.loc[len(processed_data.index)] = new_row
        #         print(processed_data.loc[len(processed_data.index) - 1].name)
        #         print((hour in processed_data.index))
        #     processed_data.at[hour, project] += curr_dur
        #     hour += pd.Timedelta('1 hour')
        #     duration -= pd.Timedelta('1 hour')

    # print(processed_data)

    """ get pca object and fit the data to it """
    pca = PCA(n_components=2)
    components = pca.fit_transform(processed_data)
    # print(components)

    """ plot with datalabels describing each row"""
    fig = px.scatter(components, x=0, y=1, hover_name=data['Hour'].unique(),
                     template=plotly.io.templates["simple_white"], labels={'0': "what", '1': "hmmm"},
                     color=pd.DatetimeIndex(data['Hour'].unique()).hour, color_continuous_scale=px.colors.cyclical.HSV,
                     size=(processed_data['Long'] + 1))

    figfig = go.Figure(data=go.Scatter(x=components[:, 0], y=components[:, 1], mode='markers',
                            text=pd.DatetimeIndex(data['Hour'].unique()),
                            marker={'color': pd.DatetimeIndex(data['Hour'].unique()).hour.astype(int),
                                    'size': ((processed_data['Long'] + 3) * 4),
                                    'colorscale': plotly.colors.cyclical.Twilight}))

    """ create text labels """
    # join columns with each row, remove long row
    annot_array = (processed_columns + ": " + processed_data.astype(str)).to_numpy()[:, :-1]

    # template for the actual labels
    text_array = np.copy(annot_array[:, 0])

    for index, element in enumerate(annot_array):
        text_array[index] = ' |'
        for idx, word in enumerate(element):
            if " 0" not in word:
                text_array[index] += "| " + word + " |"
        text_array[index] += "|"

    """ create label + coords csv file for michael """
    # new = np.vstack((pd.DatetimeIndex(data['Hour'].unique()).astype(str) + text_array,
    #                  components[:, 0], components[:, 1]))
    # pd.DataFrame(new.T).to_csv("labels_data.csv")

    figfig.update_traces(hoverinfo='text',
                         hovertext=(pd.DatetimeIndex(data['Hour'].unique()).astype(str) + text_array).tolist(),
                         selector=dict(type='scatter'))

    figfig.show()
