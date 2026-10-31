import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv('data/train.csv') #open train.csv and put it in a box called train
test = pd.read_csv('data/test.csv')

print(train.shape)
print(test.shape)

print(train.isnull().sum())   #isnull() asks of every cell “is this empty?
print(train.isnull().mean() * 100)
print(test.isnull().mean() * 100)

train['Units_Sold'] = train['Item_Outlet_Sales'] / train['Item_MRP']   #a ₹100 item with ₹1,500 in sales means 15 units were sold. Units let us compare cheap and expensive items fairly.

print(train.groupby(train['Item_Weight'].isnull())['Units_Sold'].mean())
print(train.groupby(train['Outlet_Size'].isnull())['Units_Sold'].mean())
print(train[train['Item_Weight'].isnull()]['Outlet_Identifier'].value_counts())
print(train[train['Outlet_Size'].isnull()]['Outlet_Identifier'].value_counts())    
value_counts()
            
             #train[ condition ] means “keep only the rows where this is true”, so only the rows with an empty cell.           counts which stores those rows come from.
#What it shows: empty weights come from only 2 stores, and empty sizes from only 3. “Empty” really just means “which store it is”, and we already know the store. So there’s nothing to fill in.

print(train.groupby('Outlet_Identifier')['Units_Sold'].mean())
print(train.groupby('Outlet_Type')['Units_Sold'].mean())
print(train.groupby('Item_Type')['Units_Sold'].mean())
print(train.groupby('Item_Fat_Content')['Units_Sold'].mean())

train['Price_Range'] = pd.cut(train['Item_MRP'], bins=[0, 70, 135, 200, 300])
print(train.groupby('Price_Range', observed=True)['Units_Sold'].mean())

plt.scatter(train['Item_MRP'], train['Item_Outlet_Sales'], s=3)
plt.xlabel('Item_MRP')
plt.ylabel('Item_Outlet_Sales')
plt.savefig('results/sales_vs_mrp.png')
plt.close()

q1 = train['Units_Sold'].quantile(0.25)
q3 = train['Units_Sold'].quantile(0.75)
iqr_limit = q3 + 1.5 * (q3 - q1)
sigma_limit = train['Units_Sold'].mean() + 3 * train['Units_Sold'].std()
p99_limit = train['Units_Sold'].quantile(0.99)

print('IQR limit:', iqr_limit, 'outliers:', (train['Units_Sold'] > iqr_limit).sum())
print('3 sigma limit:', sigma_limit, 'outliers:', (train['Units_Sold'] > sigma_limit).sum())
print('99th percentile limit:', p99_limit, 'outliers:', (train['Units_Sold'] > p99_limit).sum())

print(train[train['Units_Sold'] > iqr_limit]['Outlet_Identifier'].value_counts())

train['OUT010'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT010')
train['OUT013'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT013')
train['OUT017'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT017')
train['OUT018'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT018')
train['OUT019'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT019')
train['OUT027'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT027')
train['OUT035'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT035')
train['OUT045'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT045')
train['OUT046'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT046')
train['OUT049'] = train['Item_MRP'] * (train['Outlet_Identifier'] == 'OUT049')

test['OUT010'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT010')
test['OUT013'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT013')
test['OUT017'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT017')
test['OUT018'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT018')
test['OUT019'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT019')
test['OUT027'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT027')
test['OUT035'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT035')
test['OUT045'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT045')
test['OUT046'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT046')
test['OUT049'] = test['Item_MRP'] * (test['Outlet_Identifier'] == 'OUT049')

features = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
            'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049']

y = train['Item_Outlet_Sales']

train['Outlet_Number'] = train['Outlet_Identifier'].str[3:].astype(int)
X_label = train[['Item_MRP', 'Outlet_Number']]
rmse_label = -cross_val_score(LinearRegression(), X_label, y, cv=5, scoring='neg_root_mean_squared_error')
print('Label encoding RMSE:', rmse_label.mean())

X_onehot = pd.get_dummies(train['Outlet_Identifier'], dtype=float)
X_onehot['Item_MRP'] = train['Item_MRP']
rmse_onehot = -cross_val_score(LinearRegression(), X_onehot, y, cv=5, scoring='neg_root_mean_squared_error')
print('One-hot encoding RMSE:', rmse_onehot.mean())    

#get_dummies makes 10 yes/no (1/0) columns, one per store, and price is added as one column.
#Problem: price has the same effect in every store. A ₹100 item adds the same sales in a grocery store as in a big supermarket.
#Error: 1,129

X = train[features]
rmse_final = -cross_val_score(LinearRegression(fit_intercept=False), X, y, cv=5, scoring='neg_root_mean_squared_error')   

#fit_intercept=False means the line starts at 0: price 0 means sales 0.
#Error: 1,073 (best)
#line It also prints all 5 round scores (1,088, 1,067, 1,066, 1,067 and 1,075). They’re close together, so the model is stable.
             
print('MRP x Outlet encoding RMSE:', rmse_final.mean())
print('RMSE for each fold:', rmse_final)

print('RMSE if we predict the average:', y.std())

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
print(pd.Series(model.coef_, index=features))

X_test = test[features]
pred = model.predict(X_test)
pred[pred < 0] = 0

submission = pd.DataFrame()
submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']
submission['Item_Outlet_Sales'] = pred
submission.to_csv('results/submission.csv', index=False)
print(submission.shape)
