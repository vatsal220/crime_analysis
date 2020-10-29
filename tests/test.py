'''
@vatsal
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier


# constants
PATH = '/Users/vatsalpatel/Desktop/'
ASSULT_PATH = 'projects/crime_analysis/data/Assault_2014_to_2019.csv'
COLS = ['event_unique_id', 'occurrencedate', 'reporteddate', 'premisetype',
        'offence', 'reportedyear', 'reportedmonth', 'reportedday', 'reporteddayofyear',
        'reporteddayofweek', 'reportedhour', 'occurrenceyear',
       'occurrencemonth', 'occurrenceday', 'occurrencedayofyear',
       'occurrencedayofweek', 'occurrencehour', 'MCI', 'Division', 'Hood_ID',
       'Neighbourhood', 'Lat', 'Long']

# The mapping is associating the neighbourhood to the burrough
# 1 = 'Downtown Toronto', 3 = 'North York', 4 = 'Scarborough', 6 = 'Etobicoke / Missisauga'
BURROUGH_MAP_DICT = {
    'Black Creek (24)' : 1,
    'Eringate-Centennial-West Deane (11)' : 6,
    'Bendale (127)' : 4,
    'Cabbagetown-South St.James Town (71)' : 1,
    'Woburn (137)' : 3,
    'Bedford Park-Nortown (39)' : 3,
    'Mount Pleasant East (99)' : 1,
    "O'Connor-Parkview (54)" : 4,
    'Parkwoods-Donalda (45)' : 3,
    'South Riverdale (70)' : 1,
    'Danforth East York (59)' : 4,
    'Dufferin Grove (83)' : 1,
    'Eglinton East (138)' : 4,
    'Milliken (130)' : 4,
    "L'Amoreaux (117)" : 4,
    'East End-Danforth (62)' : 4,
    'Kensington-Chinatown (78)' : 1,
    'St.Andrew-Windfields (40)' : 3,
    'Banbury-Don Mills (42)' : 4,
    'Greenwood-Coxwell (65)' : 4,
    'Beechborough-Greenbrook (112)' : 4,
    'York University Heights (27)' : 3,
    'Kingsview Village-The Westway (6)' : 6,
    'Victoria Village (43)' : 3,
    'Humbermede (22)' : 6,
    'Yonge-Eglinton (100)' : 1,
    'Dorset Park (126)' : 4,
    'Waterfront Communities-The Island (77)' : 1,
    'High Park North (88)' : 6,
    'Alderwood (20)' : 6, 
    'Niagara (82)' : 1,
    'Regent Park (72)' : 1,
    'Cliffcrest (123)' : 4,
    'Birchcliffe-Cliffside (122)' : 4,
    'West Hill (136)' : 4,
    'Church-Yonge Corridor (75)' : 1,
    'Rockcliffe-Smythe (111)' : 6,
    'Junction Area (90)' : 6,
    'Mount Olive-Silverstone-Jamestown (2)' : 6,
    'Moss Park (73)' : 1,
    'Rexdale-Kipling (4)' : 6,
    'Agincourt South-Malvern West (128)' : 4,
    'Clanton Park (33)' : 3,
    'Clairlea-Birchmount (120)' : 4,
    'Humber Heights-Westmount (8)' : 6,
    'North Riverdale (68)' : 6,
    'Guildwood (140)' : 4,
    'West Humber-Clairville (1)' : 6,
    'Weston (113)' : 6,
    'University (79)' : 1,
    'Bridle Path-Sunnybrook-York Mills (41)' : 3,
    'Kennedy Park (124)' : 4,
    'Bayview Village (52)' : 3,
    'Mimico (includes Humber Bay Shores) (17)' : 6,
    'Little Portugal (84)' : 1,
    'Downsview-Roding-CFB (26)' : 3,
    'Annex (95)' : 1,
    'Newtonbrook West (36)' : 1,
    'Woodbine Corridor (64)' : 4,
    'Wexford/Maryvale (119)' : 4,
    'Rosedale-Moore Park (98)' : 1,
    "Tam O'Shanter-Sullivan (118)" : 4,
    'Taylor-Massey (61)' : 4,
    'Islington-City Centre West (14)' : 6,
    'Steeles (116)' : 3,
    'Playter Estates-Danforth (67)' : 1,
    'Forest Hill South (101)' : 1,
    'Old East York (58)' : 4,
    'Forest Hill North (102)' : 1, 
    'Flemingdon Park (44)' : 3,
    'Don Valley Village (47)' : 3,
    'Rouge (131)' : 4,
    'Oakwood Village (107)' : 1,
    'North St.James Town (74)' : 3,
    'Westminster-Branson (35)' : 3,
    'New Toronto (18)' : 1,
    'Etobicoke West Mall (13)' : 1,
    'Newtonbrook East (50)' : 3,
    'Yorkdale-Glen Park (31)' : 3,
    'Casa Loma (96)' : 1,
    'Markland Wood (12)' : 6, 
    'Trinity-Bellwoods (81)' : 1,
    'South Parkdale (85)' : 3,
    'Ionview (125)' : 4,
    'Caledonia-Fairbank (109)' : 1,
    'Woodbine-Lumsden (60)' : 4,
    'Dovercourt-Wallace Emerson-Junction (93)' : 1,
    'Highland Creek (134)' : 4,
    'Glenfield-Jane Heights (25)' : 6,
    'Humewood-Cedarvale (106)' : 6,
    'Briar Hill-Belgravia (108)' : 1,
    'Willowdale West (37)' : 3,
    'Oakridge (121)' : 4,
    'Roncesvalles (86)' : 1,
    'Bay Street Corridor (76)' : 1,
    'Broadview North (57)' : 1,
    'Runnymede-Bloor West Village (89)' : 6,
    'Willowridge-Martingrove-Richview (7)' : 6,
    'The Beaches (63)' : 4,
    'Agincourt North (129)' : 4,
    'Malvern (132)' : 6,
    'Palmerston-Little Italy (80)' : 1,
    'Morningside (135)' : 4,
    'Long Branch (19)' : 6, 
    'Weston-Pellam Park (91)' : 6,
    'Hillcrest Village (48)' : 1,
    'Stonegate-Queensway (16)' : 6,
    'Thorncliffe Park (55)' : 4,
    'Willowdale East (51)' : 3,
    'Kingsway South (15)' : 6,
    'Elms-Old Rexdale (5)' : 6,
    'Lansing-Westgate (38)' : 3,
    'Leaside-Bennington (56)' : 1,
    'Humber Summit (21)' : 6,
    'Brookhaven-Amesbury (30)' : 3,
    'Mount Dennis (115)' : 1,
    'Bayview Woods-Steeles (49)' : 4,
    'Blake-Jones (69)' : 4,
    'Rustic (28)' : 1,
    'Scarborough Village (139)' : 4,
    'Yonge-St.Clair (97)' : 1,
    'Edenbridge-Humber Valley (9)' : 6,
    'Pelmo Park-Humberlea (23)' : 6,
    'Bathurst Manor (34)' : 3,
    'Thistletown-Beaumond Heights (3)' : 6,
    'Lambton Baby Point (114)' : 6,
    'Princess-Rosethorn (10)' : 6,
    'Wychwood (94)' : 1,
    'Pleasant View (46)' : 3,
    'Danforth (66)' : 4,
    'Henry Farm (53)' : 3,
    'Keelesdale-Eglinton West (110)' : 1,
    'Lawrence Park South (103)' : 1,
    'Mount Pleasant West (104)' : 1,
    'Maple Leaf (29)' : 1,
    'High Park-Swansea (87)' : 6,
    'Englemount-Lawrence (32)' : 3,
    'Corso Italia-Davenport (92)' : 1,
    'Lawrence Park North (105)' : 6,
    'Centennial Scarborough (133)' : 4
}

sys.path.insert(1, '../src')

from crime import MCI_Crime

cf = MCI_Crime()

assult_df = pd.read_csv(PATH + ASSULT_PATH)
assult_df = assult_df[COLS]

assult_df = cf.limit_years(df = assult_df, year = 2014)
assult_df = cf.mapping_neighbourhoods(df = assult_df, mapping_dict = BURROUGH_MAP_DICT)

# cf.yearly_crimes(df = assult_df, title = 'Count of Crimes per Year - Toronto')
# cf.total_occurrences_yearly(data = assult_df, x = 'offence', hue = 'occurrenceyear', title = 'Total Offence Occurences Over Years - MCI', color = 'Blues')
# cf.total_occurrences_yearly(data = assult_df, x = 'offence', hue = 'occurrencedayofweek', title = 'Total Offence Occurences Over Day of Week', color = 'Blues')
# cf.plot_top_danger_neighbourhoods(data = assult_df, y = 'Neighbourhood', color = 'Blues', title = 'Top 10 Dangerous Neighbourhood')
# cf.plot_top_danger_neighbourhoods(data = assult_df, y = 'Burrough', color = 'Blues', title = 'Events per Mapped Neighbourhood')

feat_cols = ['feat_premisetype', 'feat_offence', 'occurrenceyear', 'feat_occurrencemonth', 'feat_occurrencedayofweek', 'occurrencehour', 'Burrough']

X,y = cf.feature_selection(data = assult_df, feat_cols = feat_cols, target_col = 'Burrough')

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35)

# create a classifier using Gradient Boost
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# predict
y_pred_GB = model.predict(X_test)
y_training_GB = model.predict(X_train)

# result display
print("Testing Accuracy:", round(metrics.accuracy_score(y_test, y_pred_GB) * 100, 2), "%")
print("Training Accuracy:", round(metrics.accuracy_score(y_train, y_training_GB)* 100, 2), "%\n")

print('Classification report: \n', metrics.classification_report(y_test, y_pred_GB), "\n")

# save the model to disk
filename = '../src/model_test_run.sav'
pickle.dump(model, open(filename, 'wb'))
print('Model Saved')
print('Test Successful')