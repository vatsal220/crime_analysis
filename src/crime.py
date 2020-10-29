import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
@vatsal
'''

class MCI_Crime():
    
    def __innit__(self):
        pass
    
    def mapping_neighbourhoods(self, df, mapping_dict):
        '''
        This function will create a new column which will map the neighbourhood to a burrough

        args:
            df (dataframe) : choose a crime related dataframe 
            mapping_dict (dict) : dictionary which will associate the locations to a value

        returns:
            the df with a new column called 'mapped_neighbourhood' which will be a numerical mapping of the neighbourhood to the burrough

        example:
            mapping_neighbourhoods(df = crime_df, mapping_dict = BURROUGH_MAP_DICT)
            '''


        df['Burrough'] = df['Neighbourhood']
        df.replace({'Burrough' : mapping_dict}, inplace = True)
        return df
    
    def limit_years(self, df, year):
        '''
        This function will limit the occurrenceyear of events to after 2014

        args:
            df (dataframe) : the crime df
            year (int) : the minimum year of events

        returns:
            df where the events are at least at the minimum year specified

        example: 
            limit_years(df = crime_df, year = 2014)
        '''
        df = df[df.occurrenceyear >= year]
        return df
    
    def yearly_crimes(self, df, title, save_plot):
        '''
        This function will count the amount of crimes occurred per year and plot it

        args:
            df (dataframe) : crime df
            title (str) : the title of the plot

        returns : 
            a plot and a dataframe with the count of crimes occurred each year

        example: 
            yearly_crimes(df = crime_df, title = 'Count of Crimes per Year - Toronto')
        '''
        yearly_crimes = pd.DataFrame()
        yearly_crimes["Occurrence_Year"] = df["occurrenceyear"].unique()
        yearly_crimes["Crime_Count"] = list(df.groupby("occurrenceyear")["occurrenceyear"].count())

        plt.clf()
        plt.style.use('ggplot')
        plt.figure(figsize = (12,6))
        sns.lineplot('Occurrence_Year', 'Crime_Count', data = yearly_crimes, color = 'steelblue')
        plt.title(title)
        plt.xticks(rotation = 45)
        plt.xlabel('Year')
        plt.ylabel('Count')
        
        if save_plot == True:
            plt.savefig('../static/images/' + title + '.jpg')
        
        plt.show()
        return yearly_crimes
    
    def total_occurrences_yearly(self, data, x, hue, title, color, save_plot):
        '''
        This function will plot the x axis in a count plot

        args:
            x, hue (str) : Names of variables in ``data`` or vector data, optional
                           Inputs for plotting long-form data. See examples for interpretation.        
            data (dataframe) : Df, array, or list of arrays, optional
                               Dataset for plotting. If ``x`` and ``y`` are absent, this is
                               interpreted as wide-form. Otherwise it is expected to be long-form.
            color (str) : The color of the bars
            title (str) : The title of the plot

       returns:
           a countplot

        example:
            total_occurrences_yearly(data = crime_df, x = 'MCI', hue = 'occurrenceyear', title = 'Total Occurences Over Years - MCI', color = 'Blues')
        '''
        plt.subplots(figsize = (12,6))
        plt.style.use('ggplot')
        sns.countplot(x = x, hue = hue, data = data, palette = color)
        plt.xticks(rotation = 45)
        plt.title(title)
        
        if save_plot == True:
            plt.savefig('../static/images/' + title + '.jpg')
        plt.show()
        
    def plot_top_danger_neighbourhoods(self, data, y, color, title, save_plot):
        '''
        This function will create a countplot on the y axis

        args:
            x, hue (str) : Names of variables in ``data`` or vector data, optional
                           Inputs for plotting long-form data. See examples for interpretation.        
            data (dataframe) : Df, array, or list of arrays, optional
                               Dataset for plotting. If ``x`` and ``y`` are absent, this is
                               interpreted as wide-form. Otherwise it is expected to be long-form.
            color (str) : The color of the bars
            title (str) : The title of the plot

        returns:
            a plot with the top 10 worst neighbourhoods

        example:
            plot_top_danger_neighbourhoods(data = crime_df, y = 'Neighbourhood', color = 'Blues', title = 'Top 10 Dangerous Neighbourhood')
        '''
        plt.clf()
        plt.style.use('ggplot')
        plt.figure(figsize = (12,6))
        sns.countplot(y = y,
                      data = data, 
                      order = data[y].value_counts().sort_values(ascending = False).iloc[:10].index, 
                      palette = sns.color_palette(color))
        plt.title(title)
        plt.xlabel('Amount of Events')
        plt.ylabel(y)
        if save_plot == True:
            plt.savefig('../static/images/' + title + '.jpg')
        plt.show()

    def feature_selection(self, data, feat_cols, target_col):
        '''
        This function will transform the input features associated to the dataset into numerical values to be able to
        pass through a model.

        args:
            data (dataframe) : the crime dataframe
            feat_cols (list) : a list of columns you wish to one hot encode to pass through to the model
            target_col (str) : the column you wish to predict based on the features selected

        returns:
            a dataframe consisting of numerical `dummies` (one hot encoded) associated to the features

        example:
            feature_selection(data = crime_df, feat_cols = ['MCI','premisetype', 'occurrenceyear' ,'occurrencemonth', 'occurrencedayofweek', 'occurrencehour', 'Burrough'], target_col = 'Burrough')
        '''
    
        MCI_MAPPING = {
        'Assault' : 1,
        'Break and Enter' : 2,
        'Robbery' : 3,
        'Theft Over' : 4,
        'Auto Theft' : 5
        }
        PREMISETYPE_MAPPING = {
            'Outside' : 1,
            'House' : 2,
            'Commercial' : 3,
            'Apartment' : 4,
            'Other' : 5
        }
        MONTH_MAPPING = {
            'January' : 1,
            'February' : 2,
            'March' : 3,
            'April' : 4,
            'May' : 5,
            'June' : 6,
            'July' : 7,
            'August' : 8,
            'September' : 9,
            'October' : 10,
            'November' : 11,
            'December' : 12
        }
        DAY_MAPPING = {
            'Monday    ' : 1,
            'Tuesday   ' : 2,
            'Wednesday ' : 3,
            'Thursday  ' : 4,
            'Friday    ' : 5,
            'Saturday  ' : 6,
            'Sunday    ' : 7
        }
#         mapping_dicts = list(MCI_MAPPING) + list(PREMISETYPE_MAPPING) + list(MONTH_MAPPING) + list(DAY_MAPPING)

        data['feat_MCI'] = data['MCI']
        data['feat_premisetype'] = data['premisetype']
        data['feat_occurrencemonth'] = data['occurrencemonth']
        data['feat_occurrencedayofweek'] = data['occurrencedayofweek']

        data.replace({'feat_MCI' : MCI_MAPPING}, inplace = True)
        data.replace({'feat_premisetype' : PREMISETYPE_MAPPING}, inplace = True)
        data.replace({'feat_occurrencemonth' : MONTH_MAPPING}, inplace = True)
        data.replace({'feat_occurrencedayofweek' : DAY_MAPPING}, inplace = True)
        data['feat_offence'] = pd.factorize(data['offence'].tolist())[0]
        data = data[feat_cols].copy()
        y = data[target_col]
        feat_df = data.copy()
        feat_df = feat_df.drop(target_col, axis = 1)

        # features selection
        X = feat_df[feat_df.columns[:]]
        return X,y
