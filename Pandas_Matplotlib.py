#pandas and matplotlib for data visualization

import numpy as np
import pandas as pd

#LESSON 2.1 - CREATING SAMPLE DATA
#creation using array
carLoans = [[1,5000,700.13,202.02,404.3, "Toyota Sienna"],
            [1,5000,700.13,202.02,404.3, "Toyota Sienna"],
            [1,5000,700.13,202.02,404.3, "Toyota Sienna"]]

colNames = ["Month",
            "Starting Balance",
            "Repayment",
            "Interest Paid",
            "Principal Paid",
            "Car Name"]

df = pd.DataFrame(data = carLoans, columns=colNames)
print(df)

#creation using numpy
carLoans = np.array([[1,5000,700.13,202.02,404.3, "Toyota Sienna"],
                     [1,5000,700.13,202.02,404.3, "Toyota Sienna"],
                     [1,5000,700.13,202.02,404.3, "Toyota Sienna"]])

colNames = ["Month",
            "Starting Balance",
            "Repayment",
            "Interest Paid",
            "Principal Paid",
            "Car Name"]

df = pd.DataFrame(data = carLoans, columns=colNames)
print(df)

#creation using dictionaries
carLoans = {"Month" : {0: 1, 1: 1, 2: 1},
            "Starting Balance" : {0: 5000, 1: 5000, 2: 5000},
            "Repayment" : {0: 700.13, 1: 700.13, 2: 700.13},
            "Interest Paid" : {0: 202.02, 1: 202.02, 2: 202.02},
            "Principal Paid" : {0: 404.3, 1: 404.3, 2: 404.3},
            "Car Name" : {0: "Toyota Sienna", 1: "Toyota Sienna", 2: "Toyota Sienna"}}

df = pd.DataFrame(data = carLoans, columns=colNames)
print(df)

#NOTE: of course don't do this in here, do it in excel and just load the sample data

#LESSON 2.3 - LOADING SAMPLE DATA
filename = "car_financing.csv" #loading csv file
df = pd.read_csv(filename) #reading the excel file
print(df)

filename = "car_financing.xlsx" #loading excel file | pip install xlrd or pip install openpyxl
df = pd.read_excel(filename) #reading the excel file
print(df)

#LESSON 2.4 - BASIC OPERATIONS FOR PANDAS
#make sure to check your file first
print(df.head()) #viewing the first 5 rows
print(df.tail()) #viewing the last 5 rows

#check the column data types
print(df.dtypes) #get the datatypes of each variable/feature
print(df.shape) #get the number of rows and columns (rows,columns)
print(df.info()) #get the column datatype and how many values are non-null

#LESSON 2.5 - SLICING
#how to select columns using double brackets (2d)
print(df[["car_type"]].head()) #select one column and first 5 rows
print(df[["car_type", "Principal Paid"]].head()) #select two columns and last 5 rows

#how to select columns using single brackets (not suggested becuz 1d)
print(df["car_type"].head()) #select one column and first 5 rows
#it is not possible to select two columns with single bracket since single bracket is one dimension only
#print(df["car_type", "Principal Paid"].head()) #this would produce an error

#PANDAS SLICING
print(df["car_type"]) #select entire car_type
print(df["car_type"][0:10]) #[start index: end index]

#selecting columns using loc (highly suggested) - this allows you to select columns, index, and slice your data
print(df.loc[:, ["car_type"]].head()) #produces the same result as above |label based means you can select by selecting the name
print(df.loc[:, "car_type"].head()) #produces the same result but this is 1d
print(df.loc[0:2, :]) #
print(df.iloc[0:2, :]) #iloc doesn't incldue the last number like in loc ex. 0:2 it would be 0:1, weird huh |integer based means you can select by selecting the index number
#LESSON 2.6 - FILTERING - selecting only certain values
print(df[["car_type"]].value_counts())

#adding filters using logical operators
car_filter = df["car_type"] == "Toyota Sienna" #produces true/false on which is toyota sienna
print(df[car_filter].head()) #getting a dataframe with only toyota sienna
print(df.loc[car_filter, :]) #just the same with heads but in here all data with toyota sienna | ";" just means that you wanted to look at all the columns
print(df[["car_type"]].value_counts()) #looking here it seems like nothing changed, well you havent assigned it back to original df tho

#assigning filtered dataframe back to the original dataframe df
df = df.loc[car_filter, :]
print(df[["car_type"]].value_counts()) #after assigning

#interest rate filter
print(df[["interest_rate"]].value_counts())
print(df["interest_rate"] == 0.0702) #producing panda series with true/false values
interest_filter = df["interest_rate"] == 0.0702
df = df.loc[interest_filter, :] #assigning all of the interest filter back to original dataframe
print(df["interest_rate"].value_counts(dropna = False)) #dropna means drop null values false

#combining filter | this does not looks beginner friendly tho
df.loc[car_filter & interest_filter, :] #in here we're using bitwise logical operators, effect of this would just be the same as applying those filters individually

#LESSON 2.7 - MANAGING COLUMNS
#renaming columns
#approach 1 is by using dictionary substitution
df = df.rename(columns = {"Starting Balance": "starting_balance",
                          "Interest Paid": "interest_paid",
                          "Principal Paid": "principal_paid",
                          "New Balance": "new_balance"})

print(df.head()) #have a look at renamed columns/features/variable

#approach 2 is by using list replacement, you need to list everything tho even if you just change one of them which is meh
df.columns = ["Month",
              "starting_balance",
              "Repayment",
              "interest_paid",
              "principal_paid",
              "new_balance",
              "term",
              "interest_rate",
              "car_type"
              ]
print(df.head()) #have a look at renamed again

#remove unnecessary columns/ deleting columns
#approach 1
df = df.drop(columns=["term"])
print(df.head())

#approach 2, i think this is the better approach
del df["Repayment"]
print(df.head())

#LESSON 2.8 - AGGREGATE FUNCTIONS
#sum the values in a column using sum() attribute
print(df["interest_paid"].sum())

#quick note: if you don't know what an attribute or method does use help()
help(df["interest_paid"].sum)

#this method gives the column datatypes + number of non-null values
print(df.info())

#LESSON 2.9 - IDENTIFYING MISSING VALUES OR NAN
print(df.info()) #find first the missing value
print(df["interest_paid"].isna().head()) #produce true/false values if missing
interest_missing = df["interest_paid"].isna()
print(df.loc[interest_missing,:]) #locate the missing value for interest_paid

#count the number of missing values, this is done by df.info() tho
print(df["interest_paid"].isna().sum())

#LESSON 2.10 - REMOVING OR FILLING MISSING DATA
print(df.info()) #find first the missing value
df[30:40].dropna(how = "any") #selecting only index 30-40, drop the entire row that contain "any" nans in them or "all"

#looking at where missing data is located
print(df["interest_paid"][30:40]) #looking only at specific indices or slicing thru them

#filling in the nan with a zero is probably a bad idea but here we go
print(df["interest_paid"][30:40].fillna(0))

#back fill in value, common with time series data
print(df["interest_paid"][30:40].fillna(method="bfill")) #use ffill for forward fill

#linear interpolation (filling in of values) fill values with the inbetween number of before and after number
print(df["interest_paid"][30:40].interpolate(method= "linear"))

interest_missing = df["interest_paid"].isna()
df.loc[interest_missing, "interest_paid"] = 93.24 #93.24 is the value we got from linear interpolation

print(df["interest_paid"].sum()) #have a look at the result now
print(df.info()) #aaand we don't have nan value anymore

#LESSON 2.11 - CONVERTING PANDAS DATAFRAMES TO NUMPY ARRAYS OR DICTIONARIES
#approach 1 - converting pandas df to numpy arrays
numArray = df.to_numpy
print(numArray)

#approach 2
numArray2 = df.values
print(numArray2)

#apprach 1 - converting pandas df to dictionaries
dictConversion = df.to_dict()
print(dictConversion)

#LESSON 2.12 - EXPORTING PANDAS DATAFRAMES TO CSV AND EXCEL FILES
#exporting pandas df to csv file
df.to_csv(path_or_buf="table_i702t60.csv",
          index = False) #index false means you don't want the indices to be included in the exportation

#exporting pandas df to excel file
df.to_excel(excel_writer="table_i702t60.xlsx",
            index = False)

#LESSON 3.1 - MATPLOTLIB BASICS
import matplotlib.pyplot as plt
import seaborn as sns #wrapper for matplotlib, to make visualizations even better

#using the knowledge we learned in pandas
filename = "table_i702t60.csv"
df = pd.read_csv(filename) #loading the data
df.head() #viewing the first 5 rows to check if something is wrong
df.info() #checking if we have some missing values as it is hard to plot datas with nans

month_number = df.loc[:, "Month"].values #df.loc[rows,columns] meaning getting all the value within rows within month column
interest_paid = df.loc[:,"interest_paid"].values
principal_paid = df.loc[:,"principal_paid"].values
print(month_number)

#we are now plotting them
plt.plot(month_number, interest_paid) #plot(x-axis, y-axis)
#plt.show()

#choosing figure style
print(plt.style.available) #show the avaialbe plot style
plt.style.use("ggplot") #use this style from r
plt.plot(month_number, interest_paid)
#plt.show()

#LESSON 3.2 - CHANGING MARKER TYPE AND COLORS
#changing marker type
plt.style.use("ggplot") #use this style from r
plt.plot(month_number, interest_paid, marker = ".", markersize = 10) #look at the new visualization it has now marker as dot there are other choices too read docu
#plt.show()

#changing color of variable/feature
plt.style.use("ggplot") #use this style from r
plt.plot(month_number, interest_paid,c = "k", marker = ".", markersize = 10) #look at the new visualization it has now marker as dot there are other choices too read docu
plt.plot(month_number, principal_paid,c = "#0000FF", marker = ".", markersize = 10) #you could change colors with hexa too
#plt.show()

#LESSON 3.3 - MATLAB STYLE VS OBJECT ORIENTED SYNTAX
#MATLAB STYLE - PREFERRED
plt.style.use("seaborn")
plt.plot(month_number, interest_paid,c = "k")
plt.plot(month_number, principal_paid,c = "k")

#OBJECT ORIENTED WAY - MEH
plt.style.use("seaborn")

x, y= (3,9)
fig, axes = plt.subplots(nrows = 1, ncols = 1) #plotting just 1 plot
axes.plot(month_number, interest_paid,c = "k")
axes.plot(month_number, principal_paid,c = "k")

#COMBINATION - cmon don't do this it's confusing
fig, axes = plt.subplots(nrows = 1, ncols = 1)
plt.plot(month_number, interest_paid, c=  "k")
axes.plot(month_number, principal_paid, c = "b")

#LESSON 3.4 - CREATING PLOT TITLES, LABELS AND LIMITS, we'll only be using matplotlib style only, although you could replace plt with axes for obj orie style
plt.style.use("seaborn")
plt.plot(month_number, interest_paid,c = "k")
plt.plot(month_number, principal_paid,c = "k")

#setting xlim and ylim
plt.xlim(left=1, right=70)
plt.ylim(bottom=0, top=1000)

#setting xlabel and ylabel
plt.xlabel("Month")
plt.ylabel("Dollars")

#setting title
plt.title("Interest and Principal Paid each Month")
#plt.show()

#changing fontsize
plt.plot(month_number, interest_paid,c = "k")
plt.plot(month_number, principal_paid,c = "k")
plt.xlabel("Month",fontsize=22)
plt.ylabel("Dollars",fontsize=22)
plt.title("Interest and Principal Paid each Month", fontsize=22)
plt.xticks(fontsize=15) #changing fontsize for xticks
plt.yticks(fontsize=15) #changing fontsize for yticks
#plt.show()

#LESSON 3.5 - GRIDS | replace plt with axes to do it with obj oriented way
plt.plot(month_number, interest_paid,c = "k")
plt.plot(month_number, principal_paid,c = "k")
plt.grid() #to obviously add grid lines
plt.grid(axis="x") #yes, it is possible to only add x axis grid lines same goes for y
plt.grid(c = "k", #changing color for gridlines
         alpha = .9, #transparency
         linestyle = "-")

#LESSON 3.6 - PLOTTING LEGENDS assist in assigning some meanings
plt.plot(month_number, interest_paid,c = "k", label = "Principal") #label is for legend
plt.plot(month_number, principal_paid,c = "k", label = "Interest")

#assigning legends | again, for doing it in obj oriented way just replace plt with axes
plt.legend(loc="center right") #y tho?
plt.legend(loc=(1.02,0)) #you could also do it this way too
plt.tight_layout() #so the legends won't be cropped
#plt.show()

#LESSON 3.7 - SAVING PLOTS TO FILES, savings plot as image
plt.plot(month_number, interest_paid,c = "k", label = "Principal") #label is for legend
plt.plot(month_number, principal_paid,c = "k", label = "Interest")
plt.xlabel("Month",fontsize=22)
plt.ylabel("Dollars",fontsize=22)
plt.title("Interest and Principal Paid each Month", fontsize=22)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc=(1.02,0), borderaxespad=0, fontsize = 20)

#to save this plot as image
plt.tight_layout() #automatically adjust so figure would fit better
plt.savefig("legendcutoff.png", dpi = 300)

#LESSON 3.8 - BOXPLOT USING PANDAS (yep pandas can be use for data visualization too) AND SEABORN
cancer_df = pd.read_csv("wisconsinBreastCancer.csv")
print(cancer_df.head())
print(cancer_df.info()) #no missing values according to this
print(cancer_df["diagnosis"].value_counts(dropna = False)) #count values for diagnosis

malignant = cancer_df.loc[cancer_df["diagnosis"] == "M", "area_mean"].values
benign = cancer_df.loc[cancer_df["diagnosis"] == "B", "area_mean"].values

#creating boxplot using matplotlib
plt.boxplot([malignant,benign], labels = ["M", "B"])
plt.show()

#creating boxplot using pandas
cancer_df.boxplot(column = "area_mean", by = "diagnosis")

#same plot but without the area_mean subtitle and title
cancer_df.boxplot(column = "area_mean", by = "diagnosis")
plt.title("")
plt.suptitle("") #why is the subtitle spelled like this??

#creating boxplot seaborn
sns.boxplot(x="diagnosis", y="area_mean", data=cancer_df)

#LESSON 4.1 - HEATMAPS - representation of data where values are depicted by colors
#sequential heatmaps - relatively low values to relatively high values (light single color to dark single color)
#categorical heatmaps - using different colors that do not have inherent ordering

#approach 1 - using seaborn with sequential colormap(highly suggested compared to matplotlib)
confusion_array = np.array([[37, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                      [0, 39,  0,  0,  0,  0,  1,  0,  2,  1],
                      [0,  0, 41,  3,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0, 44,  0,  0,  0,  0,  1,  0],
                      [0,  0,  0,  0, 37,  0,  0,  1,  0,  0],
                      [0,  0,  0,  0,  0, 46,  0,  0,  0,  2],
                      [0,  1,  0,  0,  0,  0, 51,  0,  0,  0],
                      [0,  0,  0,  1,  1,  0,  0, 46,  0,  0],
                      [0,  3,  1,  0,  0,  0,  0,  0, 44,  0],
                      [0,  0,  0,  0,  0,  1,  0,  0,  2, 44]]) #create the data
plt.figure(figsize=(6,6)) #edit some features for the figure like the figsize
sns.heatmap(confusion_array,annot = True, cmap = "Blues") #heatmap(data,annotation means if put number value,color of the map)
plt.ylabel("Actual Label") #label for y-axis, you already know this
plt.xlabel("Predicted Label")
plt.show()

#approach 2 - using seaborn with qualitative colormap
plt.figure(figsize=(6,6)) #edit some features for the figure like the figsize
sns.heatmap(confusion_array,annot = True, cmap = "Pastel1") #heatmap(data,annotation means if put number value,color of the map(this determines if it is sequential or qualitative)
plt.ylabel("Actual Label") #label for y-axis, you already know this
plt.xlabel("Predicted Label")
plt.show()

#approach 3 - using matplotlib (not suggested as it is complicated)
plt.figure(figsize=(6,6)) #editing figsize
plt.imshow(confusion_array, interpolation='nearest', cmap='Blues') #equivalent to sns.heatmap with alot of lacking features
plt.colorbar() #adding colorbar to the right
tick_marks = np.arange(10) #creating an array with range from 0-10
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10) #adding ticklabel
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout() #so the figure would fit perfectly
plt.ylabel('Actual label', size = 15) #adding label for y-axis
plt.xlabel('Predicted label', size = 15)
width, height = confusion_array.shape #getting the no. of rows and columns and assigning it to height and weight

#idk what this does
for x in range(width):
    for y in range(height):
        plt.annotate(str(confusion_array[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')
plt.show()

#LESSON 4.2 - HISTOGRAMS - almost like barplot except it deals with continous data rather than categorical and is divided by bins or interval
df = pd.read_csv("kc_house_data.csv")
df.head()

#using pandas
df["price"].head() #show the first 5 rows for price variable
df["price"].hist(bins = 30) #creating a histogram with 30 no. of bins, you could leave the parameter as blank too
plt.xticks(rotation = 90) #modifying the text on xticks by rotating them in 90 degrees
plt.show()

plt.style.use("seaborn") #use seaborn style since the default style is ugly
price_filter = df.loc[:, "price"] <= 3000000 # the data seems to be spread out because of outliers, we'll remove this from visualizing
df.loc[price_filter, "price"].hist(bins = 30, edgecolor = "black") #use the price filter
plt.show()

#LESSON 4.3 - SUBPLOTS - TO DO
#


