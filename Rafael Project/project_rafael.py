"""
import useful libraries to project
"""
try:
    import pandas as pd
    from matplotlib import pyplot as plt
    import math
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    from sklearn.model_selection import train_test_split

except ImportEmrror:
    print("Need to fix the installation")
    raise


"""
define a constant variable of maximum length route of rocket like it is in data
"""
TOTAL_SECOND_OF_ROUTE = 29


"""
define global variables for tables to easy using
"""
classification, classification_1_4_7_10 = [], []
training_set_1_16, check_set_1_16, table_1_16, training_set_1_4_7_10, check_set_1_4_7_10, table_1_4_7_10 = [], [], [], [], [], []


"""
load data as dataframe.
"""
data = pd.read_csv("./train.csv")


# $$$$$$$$$$$$$$$$$$$$$  part 1  $$$$$$$$$$$$$$$$$$$$$


# ====================================  task 1: knowing with data  ====================================


"""
use with the first column in data as index.
and delete the 'targetname' column
"""
def delete_column_targetname():
    global data
    data = data.drop(data.columns[211], axis=1)
    print("====== print data after delete the 'targetname' column ======", end='\n\n')
    print(data)


"""
display amount of routes for each type
"""
def display_amount_of_routes():
    amount_of_routes = data.groupby(['class']).agg('size')
    print("====== display amount of routes for each type ======", end='\n\n')
    print(amount_of_routes)


"""
function that calculate distance between two points
"""
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


"""
add column to data that calculate how many second have for each rocket
"""
def add_column_length():
    length = [(i - 2) / 7 - 1 for i in data.count(axis=1)]
    data["Length"] = length


"""
add column to data that calculate for each rocket the distance between first second to last second
"""
def add_column_distance():
    distances = [calculate_distance(0, 0, data.loc[i, 'posZ_0'], data.loc[i, f"posX_{int(data.loc[i, 'Length'])}"],
                                    data.loc[i, f"posY_{int(data.loc[i, 'Length'])}"],
                                    data.loc[i, f"posZ_{int(data.loc[i, 'Length'])}"]) for i in range(len(data))]
    data["Distance"] = distances


"""
display histogram of routes length for each type
"""
def display_histograms(dis):
    fig, axes = plt.subplots(5, 5, figsize=(50, 40))
    ind = 0
    for i in range(5):
        for j in range(5):
            ind += 1
            axes[i, j].bar([k for k in range(1, len(dis.loc[ind]) + 1)], dis.loc[ind])
    plt.show()


"""
histogram of routes length for each type
"""
def histograms():
    add_column_length()
    add_column_distance()
    dis = data.groupby('class').Distance.apply(list)
    display_histograms(dis)


# ====================================  task 2: drawing ====================================


"""
draw routes with 'plot' 
"""
def draw_routes(table, colors, number_of_routes):
    # relate to x and z because y is very low
    x_positions = []
    z_positions = []
    # first loop pass on number_of_routes first rockets in table that get
    # second loop pass on all seconds in route rocket
    for i in range(number_of_routes):
        for j in range(TOTAL_SECOND_OF_ROUTE):
            x_positions.append(table.loc[i, f'posX_{j}'])
            z_positions.append(table.loc[i, f'posZ_{j}'])

        plt.plot(x_positions, z_positions, color=colors[table.loc[i, 'class']])
        x_positions = []
        z_positions = []
    plt.show()


"""
draw route for first rocket by 'matplotlib' library
"""
def draw_route_for_first_rocket():
    # send the first rocket in data to draw it with color in RGB
    draw_routes(data, {data.loc[0, 'class']: [0, 0, 1]}, 1)


"""
draw routes for fifty first rocket with type 1-6 with different color for each type rocket 
"""
def draw_routes_for_fifty_first_rocket():
    # select from data only type 1-6
    six_types = data.loc[data['class'] <= 6].reset_index()
    # dictionary that define the color for each type of rocket color describe by RGB
    colors = {1: [0, 0, 1], 2: [0, 0.5, 0], 3: [1, 0, 0], 4: [0, 0.75, 0.75], 5: [0.75, 0, 0.75],
              6: [0.9290, 0.6940, 0.1250]}
    # call to display the drawing
    draw_routes(six_types, colors, 50)


"""
draw fifty first routes with type 1 and 6 with length fifteen minitues
"""
def draw_routes_for_1_6_types():
    # select from data only type 1 and 6
    types_1_6 = data.loc[((data['class'] == 1) | (data['class'] == 6)) & (data['Length'] == 15)].reset_index()
    # dictionary that define the color for each type of rocket color describe by RGB
    colors = {1: [0, 0, 1],
              6: [0.9290, 0.6940, 0.1250]}
    # call to display the drawing
    draw_routes(types_1_6, colors, 50)


# ====================================  task 3: preparing data ====================================


"""
split table of types 1 and 16 to train and check 20% for check and 80% for train
"""
def create_and_split_table_1_16():
    global check_set_1_16, training_set_1_16, table_1_16
    table_1_16 = data.loc[(data['class'] == 1) | (data['class'] == 16)].reset_index()
    check_set_1_16 = table_1_16.sample(int(len(table_1_16) * 0.2)).reset_index()
    training_set_1_16 = table_1_16.drop(table_1_16.index[check_set_1_16.index]).reset_index()

    # fill 0 instead of 'NaN' in table with types 1, 16
    table_1_16 = table_1_16.fillna(0)


"""
unpack last column 'class' of check table and save it in vector to future using
"""
def unpack_last_column():
    global classification, check_set_1_16, training_set_1_16
    classification = check_set_1_16.loc[:, 'class'].reset_index()
    check_set_1_16.drop(['class'], axis=1, inplace=True)


# ====================================  task 4: Rules-based classification ====================================


"""
draw rocket with type 1 and 16
"""
def draw_1_16_types():
    x_positions = []
    z_positions = []

    # dictionary that define the color for each type of rocket color describe by RGB
    colors = {1: [0, 0, 1],
              16: [0, 0.5, 0]}

    # first loop pass on number_of_routes rockets in training set
    # second loop pass on all seconds in route rocket
    for i in range(len(training_set_1_16)):
        for j in range(TOTAL_SECOND_OF_ROUTE):
            x_positions.append(training_set_1_16.loc[i, f'posX_{j}'])
            z_positions.append(training_set_1_16.loc[i, f'posZ_{j}'])

        # in order to display in plot type 1 separately from type 16 so insert to separately arrays
        # values in subplot is position of plot it
        if training_set_1_16.loc[i, 'class'] == 1:
            plt.subplot(1, 2, 1)
        else:
            plt.subplot(1, 2, 2)
        plt.plot(x_positions, z_positions, color=colors[training_set_1_16.loc[i, 'class']])
        plt.axis([0, 8000, 0, 25000])

        x_positions = []
        z_positions = []

    plt.show()


"""
decide if rocket is with type 1 or with type 16 by intuitive rules. the rules only from train set
"""
def intuitive_check_type(table, i, l, intuitive_result):
    # for each second in route of rocket if pos x small than 7500 and distance
    # between first second to last second small than 5000 then type is 1 else type is 16
    if (table.loc[i, f'posX_{l}'] <= 7500) and (table.loc[i, 'Distance'] <= 5000):
        intuitive_result.append(1)
    else:
        intuitive_result.append(16)
    return intuitive_result


"""
run 'intuitive_check_type' on check set and print confusion matrix and f1 score for intuitive rules
"""
def intuitive_check_type_1_or_16():
    global check_set_1_16, classification
    # draw type 1 and 16 in order to see the different
    draw_1_16_types()

    intuitive_result = []
    for i in range(len(check_set_1_16)):
        intuitive_result = intuitive_check_type(check_set_1_16, i, int(check_set_1_16.loc[i, 'Length']), intuitive_result)

    # confusion matrix and f1 score
    intuitive_check_conf_mat = confusion_matrix(classification['class'], intuitive_result)
    intuitive_check_f1_score = f1_score(list(classification['class']), intuitive_result)
    print("===== confusion matrix for check set with intuitive rules =====")
    print(intuitive_check_conf_mat)
    print(f"f1 score for check set with intuitive rules: {intuitive_check_f1_score}")


"""
calculate kinetic energy (0.5 * M=1, V=(velX, velY, velZ))
"""
def kineticEnergy(V):
    M, KineticEnergy = 1, 0
    # V is vector
    for pos in V:
        KineticEnergy += 0.5 * M * pos * pos
    return KineticEnergy


"""
calculate potential energy (M=1 * G=10 * H=pos_Z)
"""
def potentialEnergy(H):
    M = 1
    PotentialEnergy = M * 10 * H
    return PotentialEnergy


"""
define the values of plt to displaying
"""
def prepare_to_draw(energies,s1, s2, s3, title, values_axis):
    for arr in energies:
        # values in subplot is position of plot it
        plt.subplot(s1, s2, s3)
        plt.plot(arr)
        plt.title(title)
        plt.axis(values_axis)


"""
calculate energy for each second on route of each rocket with type 1 and 16
and draw energy gragh
"""
def draw_calculate_energy_1_16():
    energies1, energies16 = [], []
    # first loop pass on table with type 1 and 16
    # second loop pass on length for route of each rocket
    for i in range(len(table_1_16)):
        route = []
        for j in range(int(table_1_16.loc[i, 'Length'])):
            X = table_1_16.loc[i, f"velX_{j}"]
            Y = table_1_16.loc[i, f"velY_{j}"]
            Z = table_1_16.loc[i, f"velZ_{j}"]
            H = table_1_16.loc[i, f"posZ_{j}"]
            # energy is sum of kinetic energy and potential energy
            route.append(kineticEnergy((X, Y, Z)) + potentialEnergy(H))
        # keep energy separately for type 1 and type 16
        if table_1_16.loc[i, 'class'] == 1:
            energies1.append(route)
        else:
            energies16.append(route)
    # draw energy gragh for type 1
    prepare_to_draw(energies1, 1, 2, 1, "energy of type 1", [0, 30, 0, 350000])
    # draw energy gragh for type 16
    prepare_to_draw(energies16, 1, 2, 2, "energy of type 16", [0, 30, 0, 350000])
    plt.show()


"""
calculate energy for one second in route of rocket
"""
def calculate_one_second_energy(table, i, j):
    X = table.loc[i, f"velX_{j}"]
    Y = table.loc[i, f"velY_{j}"]
    Z = table.loc[i, f"velZ_{j}"]
    H = table.loc[i, f"posZ_{j}"]
    return kineticEnergy((X, Y, Z)) + potentialEnergy(H)


"""
decide if type of rocket is 1 or 16 with use energy
"""
def intuitive_check_type_with_energy(table, i, l, energy_result):
    # calculate energy only the end second of route rocket
    energy_at_end_second = calculate_one_second_energy(table, i, l)
    # if energy of end second of route of rocket small than 155000 then type is 1 else type is 16
    if energy_at_end_second < 155000:
        energy_result.append(1)
    else:
        energy_result.append(16)
    return energy_result


"""
run 'intuitive_check_type_with_energy' on check set and print confusion matrix and f1 score for intuitive rules with useing of energy
"""
def intuitive_check_type_1_or_16_with_energy():
    energy_result = []
    for i in range(len(check_set_1_16)):
        intuitive_check_type_with_energy(check_set_1_16, i, int(check_set_1_16.loc[i, 'Length']), energy_result)

    # confusion matrix and f1 score
    energy_check_conf_mat = confusion_matrix(list(classification['class']), energy_result)
    energy_check_f1_score = f1_score(list(classification['class']), energy_result) #  , average='micro')
    print("===== confusion matrix for check set with intuitive rules with energy =====")
    print(energy_check_conf_mat)
    print(f"f1 score for check set with intuitive rules with energy: {energy_check_f1_score}")


# ====================================  task 5: Machine learning classification ====================================


"""
use train set in order to train model of machine and check with check set
print confusion matrix and f1 score
"""
def machine_learning_on_test(model, table):
    (x_train, x_test, y_train, y_test) = train_test_split(table.loc[:, table.columns != "class"], table['class'], test_size = 0.2, random_state=0)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # calculate predict proba -> statistic array for each type
    # predict_p = model.predict_proba(x_test)
    # conf_matrix_p = confusion_matrix(y_test, np.argmax(predict_p, axis=1))
    # print(f"confusion matrix of predict proba from machine learning on test set\n", conf_matrix_p)

    conf_matrix = confusion_matrix(y_test, np.round(abs(predictions)))
    print(f"confusion matrix from machine learning on test\n",conf_matrix)
    errors_test = abs(predictions - y_test)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors_test)))
    print("Accuracy:", accuracy_score(y_test, np.round(abs(predictions))))


"""
train model RandomForestRegressor on table with type 1 and 16
"""
def machine_learning_RandomForestRegressor_1_16():
    print("machine learning: RandomForestClassifier for types 1, 16")
    machine_learning_on_test(RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs=-1), table_1_16)


"""
train model LogisticRegression on table with type 1 and 16
"""
def machine_learning_LogisticRegression_1_16():
    print("machine learning: LogisticRegression for types 1, 16")
    machine_learning_on_test(LogisticRegression(max_iter=1000, solver='liblinear'), table_1_16)


# ====================================  task 7: prepare data and classification for 4 groups ====================================


"""
split table of types 1 and 16 to train and check 20% for check and 80% for train
"""
def create_and_split_table_1_4_7_10():
    global table_1_4_7_10, check_set_1_4_7_10, training_set_1_4_7_10
    table_1_4_7_10 = data.loc[(data['class'] == 1) | (data['class'] == 4) | (data['class'] == 7) | (data['class'] == 10)].reset_index()
    check_set_1_4_7_10 = table_1_4_7_10.sample(int(len(table_1_4_7_10)*0.2)).reset_index()
    training_set_1_4_7_10 = table_1_4_7_10.drop(table_1_4_7_10.index[check_set_1_4_7_10.index]).reset_index()

    # fill 0 instead of 'NaN' in table with types 1, 4, 7, 10
    table_1_4_7_10 = table_1_4_7_10.fillna(0)


"""
unpack last column 'class' of check table and save it in vector to future using
"""
def unpack_last_column_1_4_7_10():
    global classification_1_4_7_10
    classification_1_4_7_10 = check_set_1_4_7_10.loc[:, 'class'].reset_index()
    check_set_1_4_7_10.drop(['class'], axis=1, inplace=True)


"""
draw rocket with type 1, 4, 7, 10
"""
def draw_1_4_7_10_types():
    x_positions = []
    z_positions = []

    # dictionary that define the color for each type of rocket color describe by RGB
    colors = {1: [0, 0, 1],  # blue
              4: [0, 0.5, 0],  # green
              7: [1, 0, 0],  # red
              10: [0, 0.75, 0.75]}  # light blue

    # first loop pass on number_of_routes rockets in training set
    # second loop pass on all seconds in route rocket
    for i in range(len(training_set_1_4_7_10)):
        for j in range(TOTAL_SECOND_OF_ROUTE):
            x_positions.append(training_set_1_4_7_10.loc[i, f'posX_{j}'])
            z_positions.append(training_set_1_4_7_10.loc[i, f'posZ_{j}'])

        # in order to display in plot each type separately so insert to separately arrays
        # values in subplot is position of plot it
        if training_set_1_4_7_10.loc[i, 'class'] == 1:
            plt.subplot(1, 4, 1)
            plt.title('type 1')
        elif training_set_1_4_7_10.loc[i, 'class'] == 4:
            plt.subplot(1, 4, 2)
            plt.title('type 4')
        elif training_set_1_4_7_10.loc[i, 'class'] == 7:
            plt.subplot(1, 4, 3)
            plt.title('type 7')
        else:
            plt.subplot(1, 4, 4)
            plt.title('type 10')
        plt.plot(x_positions, z_positions, color=colors[training_set_1_4_7_10.loc[i, 'class']])
        plt.axis([0, 11000, 0, 26000])

        x_positions = []
        z_positions = []

    plt.show()


"""
calculate energy for each second on route of each rocket with type 1, 4, 7, 10
and draw energy gragh
"""
def draw_calculate_energy_1_4_7_10():
    global table_1_4_7_10
    energies_1, energies_4, energies_7, energies_10 = [], [], [], []

    # first loop pass on table with type 1 and 4 and 7 and 10
    # second loop pass on length of route for each rocket
    for i in range(len(table_1_4_7_10)):
        route = []
        for j in range(int(table_1_4_7_10.loc[i, 'Length'])):
            X = table_1_4_7_10.loc[i, f"velX_{j}"]
            Y = table_1_4_7_10.loc[i, f"velY_{j}"]
            Z = table_1_4_7_10.loc[i, f"velZ_{j}"]
            H = table_1_4_7_10.loc[i, f"posZ_{j}"]

            # energy is sum of kinetic energy and potential energy
            route.append(kineticEnergy((X, Y, Z)) + potentialEnergy(H))
        # keep energy separately for each type
        if table_1_4_7_10.loc[i, 'class'] == 1:
            energies_1.append(route)
        elif table_1_4_7_10.loc[i, 'class'] == 4:
            energies_4.append(route)
        elif table_1_4_7_10.loc[i, 'class'] == 7:
            energies_7.append(route)
        else:
            energies_10.append(route)
    # draw energy gragh for type 1
    prepare_to_draw(energies_1, 1, 4, 1, 'energy of type 1', [0, 30, 0, 600000])

    # draw energy gragh for type 4
    prepare_to_draw(energies_4, 1, 4, 2, 'energy of type 4', [0, 30, 0, 600000])

    # draw energy gragh for type 7
    prepare_to_draw(energies_7, 1, 4, 3, 'energy of type 7', [0, 30, 0, 600000])

    # draw energy gragh for type 10
    prepare_to_draw(energies_10, 1, 4, 4, 'energy of type 10', [0, 30, 0, 600000])

    plt.show()


"""
decide if rocket is with type 1 or 4 or 7 or 10 by intuitive rules with calculate energy. the rules only from train set
"""
def intuitive_check_4_types(table, i, l, intuitive_result_1_4_7_10):
    # calculate energy only the end second of route rocket
    energy_at_end_point = calculate_one_second_energy(table, i, l)
    # if energy of end second of route rocket bigger than 260000 then type is 10
    if (energy_at_end_point >= 260000) or (table.loc[i, 'Distance'] >= 18160):
        intuitive_result_1_4_7_10.append(10)
    # if energy of end second of route rocket between 162500 to 260000 type is 7
    elif (energy_at_end_point >= 162500) or (table.loc[i, 'Distance'] >= 12176):
        intuitive_result_1_4_7_10.append(7)
    # if energy of end second of route rocket between 106000 to 162500 type is 4
    elif (energy_at_end_point >= 106000) or (table.loc[i, 'Distance'] >= 7365):
        intuitive_result_1_4_7_10.append(4)
    # if energy of end second of route rocket small than 106000 is 1
    else:
        intuitive_result_1_4_7_10.append(1)
    return intuitive_result_1_4_7_10


"""
run 'intuitive_check_4_types' on check set with type 1, 4, 7, 10 and print confusion matrix and f1 score for intuitive rules with energy
"""
def intuitive_check_type_1_4_7_10():
    global check_set_1_4_7_10, classification_1_4_7_10

    intuitive_result_1_4_7_10 = []
    for i in range(len(check_set_1_4_7_10)):
        intuitive_result_1_4_7_10 = intuitive_check_4_types(check_set_1_4_7_10, i, int(check_set_1_4_7_10.loc[i, 'Length']), intuitive_result_1_4_7_10)

    # confusion matrix and f1 score
    intuitive_check_conf_mat_1_4_7_10 = confusion_matrix(list(classification_1_4_7_10['class']), intuitive_result_1_4_7_10)
    intuitive_check_f1_score_1_4_7_10 = f1_score(list(classification_1_4_7_10['class']), intuitive_result_1_4_7_10, average='micro')
    print("===== confusion matrix for check set with type 1, 4, 7, 10 with intuitive rules calaulate energy =====")
    print(intuitive_check_conf_mat_1_4_7_10)
    print(f"f1 score for check set with intuitive rules: {intuitive_check_f1_score_1_4_7_10}")


"""
train model LogisticRegression on table with type 1, 4, 7, 10
"""
def machine_learning_LogisticRegression_1_4_7_10():
    print("machine learning: LogisticRegression for types 1, 4, 7, 10")
    machine_learning_on_test(LogisticRegression(max_iter=1000, solver='liblinear'), table_1_4_7_10)


"""
train model RandomForestClassifier on table with type 1, 4, 7, 10
"""
def machine_learning_RandomForestClassifier_1_4_7_10():
    print("machine learning: RandomForestClassifier for types 1, 4, 7, 10")
    machine_learning_on_test(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, max_depth=200), table_1_4_7_10)


# $$$$$$$$$$$$$$$$$$$$$  part 2  $$$$$$$$$$$$$$$$$$$$$


"""
train model RandomForestClassifier on table with type 1, 4, 7, 10 with default values
"""
def machine_learning_RandomForestClassifier_1_4_7_10_default():
    print("machine learning: RandomForestClassifier for types 1, 4, 7, 10 with default values")
    machine_learning_on_test(RandomForestClassifier(), table_1_4_7_10)


"""
use train set in order to train model of machine and check with train set
print confusion matrix and f1 score
"""
def machine_learning_on_train(model, table):
    (x_train, x_test, y_train, y_test) = train_test_split(table.loc[:, table.columns != "class"], table['class'], test_size=0.2)
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    conf_matrix = confusion_matrix(y_train, np.round(abs(predictions)))
    print(f"confusion matrix from machine learning on train set\n",conf_matrix)
    errors_test = abs(predictions - y_train)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors_test)))
    print("Accuracy:", accuracy_score(y_train, np.round(abs(predictions))))


"""
train model RandomForestClassifier on table with type 1, 4, 7, 10 with default values and check the model on train set
"""
def machine_learning_train_RandomForestClassifier_1_4_7_10_default():
    print("machine learning: RandomForestClassifier for types 1, 4, 7, 10 with default values on train set")
    machine_learning_on_train(RandomForestClassifier(), table_1_4_7_10)


"""
after I try to play in number of tree and depth of tree in RandomForestClassifier this is better with check on test set with type 1, 4, 7, 10
"""
def machine_learning_RandomForestClassifier_after_try_on_test():
    machine_learning_on_test(RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth=50), table_1_4_7_10)


"""
after I try to play in number of tree and depth of tree in RandomForestClassifier this is better with check on train set with type 1, 4, 7, 10
"""
def machine_learning_RandomForestClassifier_after_try_on_train():
    machine_learning_on_train(RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth=50), table_1_4_7_10)


# def calc_predict_proba():



if __name__ == '__main__':
    # part 1
    # task 1
    # a.
    delete_column_targetname()
    # b.
    display_amount_of_routes()
    # c.
    histograms()
    # task 2
    # a.
    draw_route_for_first_rocket()
    # b.
    draw_routes_for_fifty_first_rocket()
    # c.
    draw_routes_for_1_6_types()
    # task 3
    # a.
    create_and_split_table_1_16()
    # b.
    unpack_last_column()
    # task 4
    # a. and b.
    intuitive_check_type_1_or_16()
    # d.
    draw_calculate_energy_1_16()
    # e
    intuitive_check_type_1_or_16_with_energy()
    # task5
    # a. and b.
    machine_learning_RandomForestRegressor_1_16()
    # d.
    machine_learning_LogisticRegression_1_16()
    # task 7
    # a
    create_and_split_table_1_4_7_10()
    unpack_last_column_1_4_7_10()
    # b.
    draw_calculate_energy_1_4_7_10()
    intuitive_check_type_1_4_7_10()
    # c. d. e.
    machine_learning_LogisticRegression_1_4_7_10()
    machine_learning_RandomForestClassifier_1_4_7_10()
    # f
    draw_1_4_7_10_types()
    # part 2
    # b.
    machine_learning_RandomForestClassifier_1_4_7_10_default()
    machine_learning_train_RandomForestClassifier_1_4_7_10_default()
    machine_learning_RandomForestClassifier_after_try_on_test()
    machine_learning_RandomForestClassifier_after_try_on_train()
