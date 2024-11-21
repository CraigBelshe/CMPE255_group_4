import plotly.express as px
import pandas as pd
import plotly.io as pio
from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
census_income_kdd = fetch_ucirepo(id=117) 
  
# data (as pandas dataframes) 
X = census_income_kdd.data.features 
# y = census_income_kdd.data.targets 

# plt.figure()
# Sample data
# data = {
#     'state': ['CA', 'NY', 'TX', 'FL'],
#     'value': [10, 20, 30, 40]
# }
# df = pd.DataFrame(data)

from collections import Counter
count_dict = Counter(X['GRINST'])
us_state_to_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        "American Samoa": "AS",
        "Guam": "GU",
        "Northern Mariana Islands": "MP",
        "Puerto Rico": "PR",
        "United States Minor Outlying Islands": "UM",
        "U.S. Virgin Islands": "VI",
    }

states = []
count = []
for key in count_dict.keys():
    print(key)
    if key is np.nan:
        print('bloop')
    elif key[1:] not in ['Abroad', 'District of Columbia', 'Not in universe',]:
        print(us_state_to_abbrev[key[1:]])
        states.append(us_state_to_abbrev[key[1:]])
        count.append(count_dict[key])

print(len(states), len(count), states, count)
data = {
    'state': states,
    'frequency': count
}
df = pd.DataFrame(data)

# Create the choropleth map
fig = px.choropleth(
    df,
    locations='state',
    locationmode='USA-states',
    color='frequency',
    scope='usa',
    title='State of Previous Residence 1994-95'
)

# fig.show()
pio.write_image(fig, "choropleth_usa_map.png")