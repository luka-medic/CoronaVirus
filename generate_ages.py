import numpy as np
import pandas as pd
from scipy import stats

max_age = 100

def parse_age_groups(age_groups, max_age=max_age):
    '''
    Input:  age groups; format : ["??-??", "??-??", ..., "??+"]
    Output: age groups; format : [(??,??),(??,??), ..., (??,max_age)]
    '''
    result = []
    for string in age_groups:
        if '-' in string:
            result.append(tuple(map(int, string.split('-'))))
        if '+' in string:
            result.append((int(string[:string.index('+')]), max_age))
    return result

# Load age demography
age_demograpy = pd.read_csv(r"age_demography.csv")
age_groups_demography = parse_age_groups(age_demograpy['age.group'], max_age)

# Create age demography distributions
def generate_age_group_distribution(name, age_groups, probabilities):
    '''
    Generates discrete distribution from age demography data.
    '''
    probabilities = np.array(probabilities) / np.sum(probabilities)

    xk = np.arange(age_groups[-1][-1] + 1)  # max age
    pk = []
    for group, prob in zip(age_groups_demography, probabilities):
        a, b = group
        n = b - a + 1
        pk += n * [prob / n]

    return stats.rv_discrete(name=name, values=(xk, pk))

male_age_demograpy_distribution = generate_age_group_distribution('male_age_demograpy_distribution',age_groups_demography, age_demograpy['male'])
female_age_demograpy_distribution = generate_age_group_distribution('female_age_demograpy_distribution', age_groups_demography, age_demograpy['female'])

# Load daily acquired date for patients' age groups
day_from_first_case = 10    # first confirmed case on day 1

age_groups1 = [(0,15),(16,29),(30,49),(50,59),(60,max_age)]
age_groups2 = [(0,4),(5,14),(15,24),(25,34),(35,44),(45,54),(55,64),(65,74),(75,84),(85,max_age)]

cols1 =["age.female."+str(age_group[0])+"-"+str(age_group[1]) for age_group in age_groups1[:-1]]+["age.female."+str(age_groups1[-1][0])+"+"]+\
       ["age.male."+str(age_group[0])+"-"+str(age_group[1]) for age_group in age_groups1[:-1]]+["age.male."+str(age_groups1[-1][0])+"+"]
cols2 =["age.female."+str(age_group[0])+"-"+str(age_group[1]) for age_group in age_groups2[:-1]]+["age.female."+str(age_groups2[-1][0])+"+"]+\
       ["age.male."+str(age_group[0])+"-"+str(age_group[1]) for age_group in age_groups2[:-1]]+["age.male."+str(age_groups2[-1][0])+"+"]

data_stats1 = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",
                          index_col="date",usecols=["date"]+[col+".todate" for col in cols1],parse_dates=["date"])[18:29]
data_stats2 = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",
                          index_col="date",usecols=["date"]+[col+".todate" for col in cols2],parse_dates=["date"])[29:]

days1 = np.arange(0, len(data_stats1[cols1[0]+".todate"]))
days2 = np.arange(0, len(data_stats2[cols2[0]+".todate"]))

for col in cols1:
    data_stats1[col+".new"] = np.zeros_like(data_stats1[col+".todate"])
    data_stats1[col+".new"][0] = data_stats1[col+".todate"][0]  # First acquired day
    data_stats1[col+".new"][1:] = np.diff(data_stats1[col+".todate"])
for col in cols2:
    data_stats2[col+".new"] = np.zeros_like(data_stats2[col+".todate"])
    data_stats2[col+".new"][1:] = np.diff(data_stats2[col+".todate"])

# Interpolate between both data sets
fit_female = [0,0,0,2,5,5,4,3,2,2]
fit_male = [0,0,0,1,3,1,5,3,3,1]
for i, col in enumerate(cols2):
    if 'female' in col:
        data_stats2[col + ".new"][0] = fit_female[i]
    else:   # 'male'
        data_stats2[col + ".new"][0] = fit_male[i-len(age_groups2)]

def random_ages(distribution, a=0, b=max_age, size=1):
    '''
    Returns list of random ages between a and b ( a <= age < b ) according to chosen distribution.
    '''
    result = []
    while len(result) != size:
        age = distribution.rvs()
        if a <= age < b:
            result.append(age)
    return result

def generate_sample():
    '''
    Generates random sample of ages for new daily patients.
    '''

    sample = {'age.female.new' : [], 'age.male.new' : []}

    for day in days1:
        female_new_day = []
        male_new_day = []
        for age_group, col in zip(2*age_groups1, cols1):
            if 'female' in col:
                female_new_day += random_ages(male_age_demograpy_distribution, age_group[0], age_group[1]+1, int(data_stats1[col+".new"][day]))
            else:   # male
                male_new_day += random_ages(female_age_demograpy_distribution, age_group[0], age_group[1]+1, int(data_stats1[col+".new"][day]))
        sample['age.female.new'].append(female_new_day)
        sample['age.male.new'].append(male_new_day)
    for day in days2:
        female_new_day = []
        male_new_day = []
        for age_group, col in zip(2*age_groups2, cols2):
            if 'female' in col:
                female_new_day += random_ages(male_age_demograpy_distribution, age_group[0], age_group[1]+1, int(data_stats2[col+".new"][day]))
            else:   # male
                male_new_day += random_ages(female_age_demograpy_distribution, age_group[0], age_group[1]+1, int(data_stats2[col+".new"][day]))
        sample['age.female.new'].append(female_new_day)
        sample['age.male.new'].append(male_new_day)

    sample['date'] = data_stats1.index.union(data_stats2.index)
    sample['day'] = day_from_first_case + np.arange(0,len(sample['date']))
    sample = pd.DataFrame(data=sample)
    sample = sample[['day', 'date', 'age.female.new', 'age.male.new']]

    return sample

#%%
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# print(generate_sample())

#%% Plot
# import matplotlib.pyplot as plt
#
# plt.figure(1)
# x = np.array([0]+[el[1] for el in age_groups_demography])
# ym = np.array(age_demograpy['male']) / np.sum(age_demograpy['male'])
# yf = np.array(age_demograpy['female']) / np.sum(age_demograpy['female'])
# ym = np.append([ym[0]], ym)
# yf = np.append([yf[0]], yf)
#
# hm = male_age_demograpy_distribution.rvs(size=10000)
# hf = female_age_demograpy_distribution.rvs(size=10000)
# plt.hist(-hm, bins=np.arange(-101,2), density=True, color='b', alpha=0.5, orientation='vertical')
# h2 = plt.hist(hf, bins=np.arange(0,101), density=True, color='r', alpha=0.5, orientation='vertical')
# rescale = yf[0] / h2[0][np.nonzero(h2[0])[0][0]]
# plt.step(-x, ym/rescale, 'b', x, yf/rescale, 'r')

#%%
# import matplotlib.pyplot as plt
#
# plt.figure(2)
# x = np.array([0]+[el[1] for el in age_groups_demography])
# ym = np.array(age_demograpy['male']) / np.sum(age_demograpy['male'])
# yf = np.array(age_demograpy['female']) / np.sum(age_demograpy['female'])
# ym = np.append([ym[0]], ym)
# yf = np.append([yf[0]], yf)
#
#
# hm = np.array(random_ages(male_age_demograpy_distribution, 10, 50, 10000))
# hf = np.array(random_ages(female_age_demograpy_distribution, 10, 50, 10000))
# plt.hist(-hm, bins=np.arange(-101,2), density=True, color='b', alpha=0.5, orientation='vertical')
# h2 = plt.hist(hf, bins=np.arange(0,101), density=True, color='r', alpha=0.5, orientation='vertical')
# rescale = yf[0] / h2[0][np.nonzero(h2[0])[0][0]]
# plt.step(-x, ym/rescale, 'b', x, yf/rescale, 'r')
#
# plt.show()