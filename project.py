import statsmodels.formula.api as model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv


def read_file(path):
    """
    Here we read usual data
    """

    data = []
    with open(path, encoding="UTF-8") as file:
        for line in file:
            line = line.split(",")
            data.append(line)
    x = data.pop(0)
    vals_home = ['HomeTeam', 'FTHG', 'FTR', 'HTHG',
                 'HTR', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']
    vals_away = ['AwayTeam', 'FTAG', 'FTR', 'HTAG',
                 'HTR', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']
    indexes_for_home = {}
    indexes_for_away = {}
    for i in range(len(x)):
        if x[i] in vals_home:
            indexes_for_home[x[i]] = i

    for i in range(len(x)):
        if x[i] in vals_away:
            indexes_for_away[x[i]] = i

    result = {}
    for line in data:

        if 'HomeTeam' + line[indexes_for_home[vals_home[0]]] not in result:
            result['HomeTeam' + line[indexes_for_home[vals_home[0]]]] = []
        if 'AwayTeam' + line[indexes_for_away[vals_away[0]]] not in result:
            result['AwayTeam' + line[indexes_for_away[vals_away[0]]]] = []
        home_list = []

        for val in vals_home:
            if val != 'HomeTeam':
                home_list.append(line[indexes_for_home[val]])

        result['HomeTeam' + line[indexes_for_home[vals_home[0]]]
               ].append(home_list)

        away_list = []
        for val in vals_away:
            if val != 'AwayTeam':
                away_list.append(line[indexes_for_away[val]])

        result['AwayTeam' + line[indexes_for_away[vals_away[0]]]
               ].append(away_list)

    return result


def read_file_for_prediction(path):
    """
    Here we read half of data from the next year to test our model
    """
    data = []
    with open(path, encoding="UTF-8") as file:
        for line in file:
            line = line.split(",")
            data.append(line)
    x = data.pop(0)
    vals_home = ['HomeTeam', 'FTHG', 'FTR', 'HTHG',
                 'HTR', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']
    vals_away = ['AwayTeam', 'FTAG', 'FTR', 'HTAG',
                 'HTR', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']
    indexes_for_home = {}
    indexes_for_away = {}
    for i in range(len(x)):
        if x[i] in vals_home:
            indexes_for_home[x[i]] = i

    for i in range(len(x)):
        if x[i] in vals_away:
            indexes_for_away[x[i]] = i

    result = {}
    for line in data:

        if 'HomeTeam' + line[indexes_for_home[vals_home[0]]] not in result:
            result['HomeTeam' + line[indexes_for_home[vals_home[0]]]] = []
        if 'AwayTeam' + line[indexes_for_away[vals_away[0]]] not in result:
            result['AwayTeam' + line[indexes_for_away[vals_away[0]]]] = []
        home_list = []

        for val in vals_home:
            if val != 'HomeTeam':
                home_list.append(line[indexes_for_home[val]])
        if len(result['HomeTeam' + line[indexes_for_home[vals_home[0]]]]) < 20:
            result['HomeTeam' + line[indexes_for_home[vals_home[0]]]
                   ].append(home_list)

        away_list = []
        for val in vals_away:
            if val != 'AwayTeam':
                away_list.append(line[indexes_for_away[val]])
        if len(result['AwayTeam' + line[indexes_for_away[vals_away[0]]]]) < 20:
            result['AwayTeam' + line[indexes_for_away[vals_away[0]]]
                   ].append(away_list)

    return result


def avarage(team, list_2d):
    """
    This function finds avarage values for each parameter team earns during the league
    """
    result = []
    for idx in range(len(list_2d[0])):
        avg_val = 0

        avg_win = 0
        avg_draw = 0
        avg_loose = 0

        for sub_list in list_2d:
            try:
                avg_val += int(sub_list[idx])
            except ValueError:
                if 'HomeTeam' in team:
                    if sub_list[idx] == 'H':
                        avg_win += 1
                    elif sub_list[idx] == 'D':
                        avg_draw += 1
                    elif sub_list[idx] == 'A':
                        avg_loose += 1
                else:
                    if sub_list[idx] == 'H':
                        avg_loose += 1
                    elif sub_list[idx] == 'D':
                        avg_draw += 1
                    elif sub_list[idx] == 'A':
                        avg_win += 1

        if avg_win + avg_draw + avg_loose != 0:
            avg_win /= len(list_2d)
            result.append(avg_win)
            avg_draw /= len(list_2d)
            result.append(avg_draw)
            avg_loose /= len(list_2d)
            result.append(avg_loose)
        else:
            avg_val /= len(list_2d)
            result.append(avg_val)
    return result


def find_avg_for_all_teams(dict_teams):
    """
    This function applies avarage() to every team in the list
    """
    result = {}
    for team in dict_teams:
        result[team] = avarage(team, dict_teams[team])
    return result


def create_true_table(data):
    """
    This function creates real table of laeders for the given year. (we can check in the internet)
    """
    real_table = {}
    for team in data:
        wins = 0
        draws = 0
        looses = 0
        for val in data[team]:
            if 'HomeTeam' in team:
                if val[1] == 'H':
                    wins += 1
                elif val[1] == 'D':
                    draws += 1
                elif val[1] == 'A':
                    looses += 1
            else:
                if val[1] == 'H':
                    looses += 1
                elif val[1] == 'D':
                    draws += 1
                elif val[1] == 'A':
                    wins += 1
        if team[8:] not in real_table:
            real_table[team[8:]] = [wins, draws, looses]
        else:
            real_table[team[8:]][0] += wins
            real_table[team[8:]][1] += draws
            real_table[team[8:]][2] += looses

    for team in real_table:
        real_table[team] = real_table[team][0] * 3 + real_table[team][1]

    result = sorted(list(real_table.items()), key=lambda x: x[-1])
    return result[::-1]


def home_away_to_usual(data):
    """
    This function converts data to more comfortable form
    """
    result = {}
    for team in data:
        if team[8:] not in result:
            result[team[8:]] = data[team]
        else:
            for i in range(len(data[team])):
                result[team[8:]][i] += data[team][i]
    return result


def write_data_for_ols(data):
    """
    Here we write down data from the previous year into a file for farther usage in ols model
    """
    real_table = create_true_table(data)
    avg_results_for_league = home_away_to_usual(find_avg_for_all_teams(data))

    header = ['TeamName', 'LeaguePlace', 'FullTimeGoals', 'FullTimeWin', 'FullTimeDraw', 'FullTimeLoose', 'HalfTimeGolas', 'HalfTimeWin',
              'HalfTimeDraw', 'HalfTimeLoose', 'shots', 'shots on target', 'fals', 'corners', 'yeallow_card', 'red_card']

    data = []
    for i in range(len(real_table)):
        data.append([real_table[i][0], i + 1] +
                    avg_results_for_league[real_table[i][0]])

    with open("csv_for_ols.csv", 'w', encoding='UTF8', newline='') as file:
        file.truncate()
        writer = csv.writer(file)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


def check_for_corelation():
    """
    This function checks corellation of LeaguePlace with other values.
    Also it checks corellation between all the variables.
    """
    df = pd.read_csv("csv_for_ols.csv")
    print('\nCorelation')
    print(df.corr()['LeaguePlace'])

    matrix = df[['FullTimeGoals', 'FullTimeWin', 'FullTimeDraw', 'FullTimeLoose', 'HalfTimeGolas', 'HalfTimeWin',
                 'HalfTimeDraw', 'HalfTimeLoose', 'shots', 'shots on target', 'fals', 'corners', 'yeallow_card', 'red_card']].corr().round(2)

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, vmax=1, vmin=-
                1, center=0, cmap='vlag', mask=mask)
    plt.show()

    matrix = df[['FullTimeDraw', 'HalfTimeGolas', 'HalfTimeDraw', 'fals',
                 'yeallow_card', 'red_card']].corr().round(2)

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, vmax=1, vmin=-
                1, center=0, cmap='vlag', mask=mask)
    plt.show()


def make_model():
    print('\nOLS model')
    df = pd.read_csv("csv_for_ols.csv")
    place_pred = model.ols(
        formula='LeaguePlace ~ HalfTimeGolas + HalfTimeDraw + red_card', data=df).fit()

    print(place_pred.summary())



if __name__ == "__main__":
    data = read_file_for_prediction("2019.csv")
    data_prev = read_file("2018.csv")

    write_data_for_ols(data_prev)
    check_for_corelation()
    make_model()

    data = home_away_to_usual(find_avg_for_all_teams(data))

    result = []
    for team in data:
        result.append([team, 31.9380 + (-10.1212) * data[team][4] + (-10.3967) *
                    data[team][6] + (-5.3477) * data[team][-1]])
    result.sort(key=lambda x: x[-1])

    print('\nPredicted table')
    for i in result:
        print(i)
