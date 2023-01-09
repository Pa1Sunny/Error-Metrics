import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import create_engine

class Assignment:
    def init(self):
        pass
    def Search_ideal_match(self, train_func, ideal_func):
        """
        function finds matches between training functions and ideal functions based on 
         min(MSE)
          :param train_func: define training functions
           :param ideal_func: define ideal functions set
            :return: ideal functions dataframe and their deviations
        """
# find last parameters of both fucntions
        if isinstance(train_func, pd.DataFrame) and isinstance(ideal_func, pd.DataFrame):
            ideal_lcol = len(ideal_func.columns)
            train_lrow = train_func.index[-1] + 1
            train_col = len(train_func.columns)
            
# Loop and find perfect four functions
            index_list = [] 
            least_square = []
            for j in range(1, train_col): 
                least_square1 = []
                for k in range(1, ideal_lcol): 
                    MSE_sum = 0
                    for i in range(train_lrow): 
                        z1 = train_func.iloc[i, j] 
                        z2 = ideal_func.iloc[i, k] 
                        MSE_sum = MSE_sum + ((z1 - z2) ** 2)
                    least_square1.append(MSE_sum / train_lrow)
                min_least = min(least_square1)
                index = least_square1.index(min_least) 
                index_list.append(index + 1)
                least_square.append(min_least)

            per_frame = pd.DataFrame(list(zip(index_list, least_square)), columns=["Index", "least_square_value"])

            return per_frame
        else:
            raise TypeError("Given arguments are not of Dataframe type.")
    def find_ideal_via_row(self, test_func):
        """
        determine for each and every x-y-pair of values whether they can be assigned 
         to the four chosen ideal functions
          :param test_func: Dataframe with x and y values
           :return: test function paired with values from the four ideal functions
        """
        if isinstance(test_func, pd.DataFrame):
            test_lrow = test_func.index[-1] + 1
            test_lcol = len(test_func.columns) 
            ideal_index = [] 
            deviation = [] 
            for j in range(test_lrow): 
                MSE_l = [] 
                for i in range(2, test_lcol): 
                    z1 = test_func.iloc[j, 1]
                    z2 = test_func.iloc[j, i]
                    MSE_sum = ((z2 - z1) ** 2) 
                    MSE_l.append(MSE_sum) 
                min_least = min(MSE_l) 
                if min_least < (np.sqrt(2))*0.08:
                    deviation.append(min_least) 
                    index = MSE_l.index(min_least) 
                    ideal_index.append(index) 
                else:
                    deviation.append(min_least)
                    ideal_index.append("Miss")


                test["Deviation"] = deviation
                test["Ideal index"] = ideal_index

                return test

            else:
                raise TypeError("Given argument is not of Dataframe type.")
            
    def prepare_graphs(self, x_func, x_par, y1_func, y1_par, y2_func, y2_par, show_plots=True):
        """
        function prepares a plot based on given paramaters
         :param x_func: x function
          :param x_par: x position
           :param y1_func: y1 function
            :param y1_par: y1 position
             :param y2_func: y2 function
              :param y2_par: y2 position
               :param show_plots: True/False to display plot
                :return: graph of x and y
        """
        x = x_func.iloc[:, x_par] 
        y1 = y1_func.iloc[:, y1_par] 
        y2 = y2_func.iloc[:, y2_par] 
        plt.plot(x, y1, c="r", label="Train function") 
        plt.plot(x, y2, c="b", label="Ideal function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=3)
        if show_plots is True:
            plt.show() 
            plt.clf() 
        elif show_plots is False:
            pass
        else:
            pass
class SqliteDb(Assignment):
    """
    Load data into Sqlite database
    """
    def db_and_table_creation(self, dataframe, db_name, table_name):
        """
        function creates a database from a dataframe input
         :param dataframe: dataframe
          :param db_name: database name
           :param table_name: table name
            :return: database file into the same folder as the project
        """
        try:
            engine = create_engine(f"sqlite:///{db_name}.db", echo=True) 
            sqlite_connection = engine.connect() 
            for i in range(len(dataframes)): 
                dataframez = dataframe[i]
                dataframez.to_sql(table_name[i], sqlite_connection, if_exists="fail") 
            sqlite_connection.close() 
        except Exception:
            exception_type, exception_value, exception_traceback = sys.exc_info() 
            print(exception_type, exception_value, exception_traceback)

# read CSV files and load them into Dataframes
train = pd.read_csv("train.csv")
ideal = pd.read_csv("ideal.csv")
test = pd.read_csv("test.csv")

# get ideal functions based on train data
df = Assignment().Search_ideal_match(train, ideal)
print(df)

# plot graph of all 4 pair functions together
graph = Assignment()
for i in range(1, len(train.columns)):
    graph.prepare_graphs(train, 0, train, i, ideal, df.iloc[i-1, 0], False)

# Clean test df
test = test.sort_values(by=["x"], ascending=True) 
test = test.reset_index() 
test = test.drop(columns=["index"])

# Get x, y values of each of the 4 ideal functions
ideals = []
for i in range(0, 4):
    ideals.append(ideal[["x", f"y{str(df.iloc[i, 0])}"]])

# merge test and 4 ideal functions
for i in ideals:
    test = test.merge(i, on="x", how="left")

# determine for each and every x-y-pair of values whether or not they can be assigned to the four chosen ideal functions
test = Assignment().find_ideal_via_row(test)

# Replace values with ideal function names
for i in range(0, 4):
    test["Ideal index"] = test["Ideal index"].replace([i], str(f"y{df.iloc[i, 0]}"))

# add y values to another test_func (used later for scatter plot)
test_scat = test
test_scat["ideal y value"] = ""
for i in range(0, 100):
    k = test_scat.iloc[i, 7]
if k == "y34":
    test_scat.iloc[i, 8] = test_scat.iloc[i, 2]
elif k == "y11":
    test_scat.iloc[i, 8] = test_scat.iloc[i, 3]
elif k == "y43":
    test_scat.iloc[i, 8] = test_scat.iloc[i, 4]
elif k == "y23":
    test_scat.iloc[i, 8] = test_scat.iloc[i, 5]
elif k == "Miss":
    test_scat.iloc[i, 8] = test_scat.iloc[i, 1]

# rename columns for the train table
train = train.rename(columns={"y1": "Y1 (training funcc)", "y2": "Y2 (training funcc)",
                               "y11": "Y3 (training funcc)", "y4": "Y4 (training funcc)"})

# rename columns for the ideal table
for col in ideal.columns: 
    if len(col) > 1: 
        ideal = ideal.rename(columns={col: f"{col} (ideal funcc)"})

# rename columns for the test table
test = test.rename(columns={"x": "X (test funcc)",
                             "y": "Y (test funcc)",
                              "Deviation": "Delta Y (test funcc)",
                               "Ideal index": "No. of ideal funcc"})

# Load data to sqlite
dbs = SqliteDb()
dataframes = [train, ideal, test]
table_names = ["train_table", "ideal_table", "test_table"]
dbs.db_and_table_creation(dataframes, "assignment_database", table_names)

# Visualization
# train functions
plt.clf()
x = train.iloc[:, 0]
for i in range(1, len(train.columns)):
    plt.plot(x, train.iloc[:, i], c="g", label=f"Train function y{i}")
    plt.legend(loc=3)
    plt.show()
    plt.clf()

# ideal functions (4 chosen)
plt.clf()
x = train.iloc[:, 0]
for i in range(0, df.index[-1] + 1):
    y = df.iloc[i, 0] 
    plt.plot(x, ideal.iloc[:, y], c="#FF4500", label=f"Ideal function y{y}")
    plt.legend(loc=3)
    plt.show()
    plt.clf()
# test scatter (show points of test.csv)
plt.clf() 
plt.scatter(test.iloc[:, 0], test.iloc[:, 1]) 
plt.show()
plt.clf() 
x1 = []
x2 = []
x3 = []
x4 = []
xm = []
y1 = []
y2 = []
y11 =[]
y4 = []
ym = []
# append x and y values to the upper lists
for i in range(0, 100):
    k = test_scat.iloc[i, 7]
if k == "y34":
    x1.append(test_scat.iloc[i, 0]) 
    y1.append(test_scat.iloc[i, 8]) 
elif k == "y11":
    x2.append(test_scat.iloc[i, 0]) 
    y2.append(test_scat.iloc[i, 8]) 
elif k == "y43":
    x3.append(test_scat.iloc[i, 0]) 
    y11.append(test_scat.iloc[i, 8]) 
elif k == "y23":
    x4.append(test_scat.iloc[i, 0]) 
    y4.append(test_scat.iloc[i, 8]) 
elif k == "Miss":
    xm.append(test_scat.iloc[i, 0]) 
    ym.append(test_scat.iloc[i, 8]) 
# plot ideal functions and test y-values on the same scatter plot
plt.scatter(x1, y1, marker="o", label="Test - y34", color="r")
plt.scatter(x2, y2, marker="s", label="Test - y11", color="b")
plt.scatter(x3, y11, marker="^", label="Test - y43", color="g")
plt.scatter(x4, y4, marker="d", label="Test - y23", color="#FFD700")
plt.scatter(xm, ym, marker="x", label="Test - Miss", color="#000000")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 34], label="Ideal - Y34", color="#FA8072")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 11], label="Ideal - Y11", color="#1E90FF")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 43], label="Ideal - Y43", color="#7CFC00")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 23], label="Ideal - Y23", color="#FFA500")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()
