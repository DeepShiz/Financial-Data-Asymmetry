import math
import pandas as pd
import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import ta

class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data
        self.rsi = None
        self.mfi = None
        self.roc = None
        self.cci = None
        self.apr = None
        self.anr = None
        self.test_data = None
        self.val_data = None

    def ROC(self, n):
        '''
        n = number of days for which you
        want to calculate ROC.
        The ROC function is used to calculate the
        financial ROC using past n days' data.
        '''

        N = self.data['Close'].diff(n)
        D = self.data['Close'].shift(n)
        self.data['ROC'] = pd.Series(N/D)
        return self.data

    def CCI(self, n):
        '''
        n = number of days for which you want
        to calculate CCI
        The CCI function can be used to calculate the
        CCI over past n days' data
        '''
        TP = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['CCI']= pd.Series((TP - TP.rolling(window=n, center = False).mean()) / (0.015 * TP.rolling(window=n, center=False).std()))
        return self.data

    def technical_analysis(self):
        '''
        The technical analysis function can be used
        to compute 4 technical analysis measures
        together namely, MFI, RSI, ROC and CCI and adds
        them to the dataframe passed through constructor

        Return Type = pd.DataFrame
        '''
        self.mfi = ta.volume.money_flow_index(self.data['High'], self.data['Low'], self.data['Close'], self.data['Total Trade Quantity'],n=14)
        self.rsi = ta.momentum.rsi(self.data['Close'])
        self.roc = self.ROC(n=126)
        self.cci = self.CCI(n=20)
        self.data['MFI'] = self.mfi
        self.data['RSI'] = self.rsi

        return self.data

    def add_past_closes(self):
        '''
        Function to add past 21 days closes to the dataframe
        passed through constructor

        Return type : pd.DataFrame
        '''
        closes = self.data['Close'].tolist()
        #print(closes)
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[], [], [], [], []

        for i in range(0,len(closes)):
            try:
                c21 += [closes[i-21]]
                c1 += [closes[i-1]]
                c2 += [closes[i-2]]
                c3 += [closes[i-3]]
                c4 += [closes[i-4]]
                c5 += [closes[i-5]]
                c6 += [closes[i-6]]
                c7 += [closes[i-7]]
                c8 += [closes[i-8]]
                c9 += [closes[i-9]]
                c10 += [closes[i-10]]
                c11 += [closes[i-11]]
                c12 += [closes[i-12]]
                c13 += [closes[i-13]]
                c14 += [closes[i-14]]
                c15 += [closes[i-15]]
                c16 += [closes[i-16]]
                c17 += [closes[i-17]]
                c18 += [closes[i-18]]
                c19 += [closes[i-19]]
                c20 += [closes[i-20]]
            except:
                c1 += [0]
                c2 += [0]
                c3 += [0]
                c4 += [0]
                c5 += [0]
                c6 += [0]
                c7 += [0]
                c8 += [0]
                c9 += [0]
                c10 += [0]
                c11 += [0]
                c12 += [0]
                c13 += [0]
                c14 += [0]
                c15 += [0]
                c16 += [0]
                c17 += [0]
                c18 += [0]
                c19 += [0]
                c20 += [0]
                c21 += [0]
                continue
        self.data['C1'], self.data['C2'], self.data['C3'], self.data['C4'], self.data['C5'], self.data['C6'], self.data['C7'],self.data['C8'], self.data['C9'], self.data['C10'], self.data['C11'], self.data['C12'], self.data['C13'], self.data['C14'],self.data['C15'], self.data['C16'], self.data['C17'], self.data['C18'], self.data['C19'], self.data['C20'], self.data['C21'] = c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21

        return self.data

    def calculate_returns(self):
        '''
        The average positive and negative returns are calculated for each
        day in the dataset taking into consideration the past 21 days' data
        It then adds 2 columns to the dataframe passed through constructor
        namely, 'AvgNegRet' and 'AvgPosRet'

        Return type : pd.DataFrame
        '''
        data = self.add_past_closes()
        avg_pos_return = [0]*21
        avg_neg_return = [0]*21
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21 = self.data['C1'], self.data['C2'], self.data['C3'], self.data['C4'], self.data['C5'], self.data['C6'], self.data['C7'],self.data['C8'], self.data['C9'], self.data['C10'], self.data['C11'], self.data['C12'], self.data['C13'], self.data['C14'],self.data['C15'], self.data['C16'], self.data['C17'], self.data['C18'], self.data['C19'], self.data['C20'], self.data['C21']

        for i in range(21, len(c1)):
            pos_return = 0
            neg_return = 0
            p_count = 0
            n_count = 0
            row_close = []
            returns = []
            row = [c1[i]] + [c2[i]] + [c3[i]] + [c4[i]] + [c5[i]] + [c6[i]] + [c7[i]] + [c8[i]] + [c9[i]] + [c20[i]] + [c11[i]] + [c12[i]] + [c13[i]] + [c14[i]] + [c15[i]] + [c16[i]] + [c17[i]] + [c18[i]] + [c19[i]] + [c20[i]] + [c20[i]]
            returns = [row[1]-row[0], row[2]-row[1], row[3]-row[2], row[4]-row[3], row[5]-row[4], row[6]-row[5], row[7]-row[6], row[8]-row[7], row[9]-row[8], row[10]-row[9], row[11]-row[10], row[12]-row[11], row[13]-row[12], row[14]-row[13], row[15]-row[14], row[16]-row[15], row[17]-row[16], row[18]-row[17], row[19]-row[18], row[20]-row[19]]
            for i in range(0,len(returns)):
                if(returns[i]>=0):
                    pos_return += returns[i]
                    p_count += 1
                else:
                    neg_return += returns[i]
                    n_count += 1
            avg_pos_return += [pos_return/p_count]
            avg_neg_return += [neg_return/n_count]
        self.data['AvgPosRet'] = avg_pos_return
        self.data['AvgNegRet'] = avg_neg_return

        return self.data

    def calculate_cummulative_avg_returns(self):
        '''
        The cummulative average returns are calculated for each of
        the 2 columns, AvgNegRet and AvgPosRet
        The sum of the columns is taken and divided by the number of elements
        to get the values for apr and anr

        Return Types : float64, float64
        '''
        self.calculate_returns()
        self.data['AvgNegRet'].fillna((self.data['AvgNegRet'].mean()), inplace=True)
        self.data['AvgPosRet'].fillna((self.data['AvgPosRet'].mean()), inplace=True)
        nr = self.data['AvgNegRet'].tolist()
        pr = self.data['AvgPosRet'].tolist()
        self.apr = sum(pr[21:])/(len(pr)-21)
        self.anr = sum(nr[21:])/(len(nr)-21)
        print(self.apr, self.anr)

        return self.apr, self.anr

    def add_next_close(self):
        '''
        Function to add the next day's actual close rate

        Return Type : pd.DataFrame
        '''
        next_close = self.data['Close'].tolist()
        next_close = next_close[1:] + [0]
        self.data['Sn+1'] = next_close
        #print(self.data.columns)
        return self.data

    def get_close_difference(self):
        '''
        Function that calculates difference between current
        day's closing and next day's closing for historic data

        Return type : pd.DataFrame
        '''
        diff = []
        close_n = self.data['Close'].tolist()
        close_n1 = self.data['Sn+1'].tolist()
        for i in range(0,len(close_n)):
            diff += [close_n1[i]-close_n[i]]
        self.data['Difference'] = diff
        return self.data

    def get_complete_data(self):
        '''
        A wrapper function that invokes all necessary functions
        and adds necessary columns to the dataset passed through
        constructor to make the data ready to pass through a model

        Return type : pd.DataFrame
        '''
        self.technical_analysis()
        self.calculate_returns()
        self.add_next_close()
        self.get_close_difference()
        return self.data

    def calculate_risk(self, p, n):
        '''
        p = Average Positive Return (list)
        n = Average Negative Return (list)

        Calculates the difference between the absolute values
        of p and n for each day and then checks whether the stock
        is volatile/risky to buy

        Return Type : list
        '''
        risk = []
        for i in range(0, len(p)):
            if((p[i]-n[i]) > 10):
                risk += [1]
            else:
                risk += [0]

        return risk

    def risk_split(self, inp, op, pred_gbr, pred_rfr):
        p = inp['AvgPosRet'].tolist()
        n = inp['AvgNegRet'].tolist()
        risk = self.calculate_risk(p,n)
        inp['Sn+1'] = op
        inp['Risk'] = risk
        op['Risk'] = risk
        inp['S^n+1g'] = pred_gbr
        inp['S^n+1r'] = pred_rfr
        inp_low = inp[inp['Risk']==0]
        op_low = op[op['Risk']==0]
        inp_high = inp[inp['Risk']==1]
        op_high = op[op['Risk']==1]
        op_low = op_low.drop(columns=['Risk'])
        op_high = op_high.drop(columns=['Risk'])

        return inp_low, op_low, inp_high, op_high

    def get_fpm_after_risk(self, x):
        close_n = x['Close'].tolist()
        close_n1 = x['S^n+1r'].tolist()
        nr_low = x['AvgNegRet'].tolist()
        pr_low = x['AvgPosRet'].tolist()
        anr = sum(nr_low)/len(nr_low)
        apr = sum(pr_low)/len(pr_low)
        F_NR, P_NR, M_NR, F_HR, P_HR, M_HR = [], [], [], [], [], []
        #print(len(close_n), len(close_n1))
        for i in range(0,len(close_n)):
            if(close_n1[i]-close_n[i] > 0):
                F_NR += [close_n1[i]-close_n[i]]
                P_NR += [apr*close_n[i]]
                M_NR += [anr*close_n[i]]
            else:
                F_NR += [close_n1[i]-close_n[i]]
                P_NR += [anr*close_n[i]]
                M_NR += [apr*close_n[i]]
        x['F'] = F_NR
        x['P'] = P_NR
        x['M'] = M_NR
        diff = []
        close_n = x['Close'].tolist()
        close_n1 = x['Sn+1'].tolist()
        for i in range(0,len(close_n)):
            diff += [close_n1[i]-close_n[i]]
        x['Difference'] = diff

        return x


    def ta_pred_model(self, data):
        data = data[np.isfinite(data['ROC'])]
        data = data[np.isfinite(data['AvgPosRet'])]
        data = data[np.isfinite(data['AvgNegRet'])]
        inp = data[['High','Open','Low','Close','Total Trade Quantity','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','AvgPosRet', 'AvgNegRet', 'MFI','RSI']]
        op = data[['Sn+1']]

        train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.20, random_state = 999)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.20, random_state=999)

        gbr = GradientBoostingRegressor().fit(train_x, train_y)	#Fitting and creating a model
        rfr = RandomForestRegressor().fit(train_x, train_y)

        #joblib.dump(gbr, 'gbr.pkl')
        #joblib.dump(rfr, 'With_ta.pkl')
        pred_rfr = rfr.predict(test_x)		#Predicting the answers for valdiation data
        mse_rfr = mean_squared_error(pred_rfr, test_y)	#finding the mean squared error
        r2_model_1 = r2_score(pred_rfr, test_y)


        pred_gbr = gbr.predict(test_x)
        mse_gbr = mean_squared_error(pred_gbr, test_y)

        pred_gbrv = gbr.predict(val_x)
        pred_rfrv = rfr.predict(val_x)

        r2_score_val = r2_score(pred_rfrv, val_y)
        test_x_low, test_y_low, test_x_high, test_y_high = self.risk_split(test_x, test_y, pred_gbr, pred_rfr)
        val_x_low, val_y_low, val_x_high, val_y_high = self.risk_split(val_x, val_y, pred_gbrv, pred_rfrv)

        '''
        Calculate the FPM values for both
        the models, i.e. GBR and RFR.
        '''

        test_x_low = self.get_fpm_after_risk(test_x_low)
        test_x_high = self.get_fpm_after_risk(test_x_high)
        val_x_low = self.get_fpm_after_risk(val_x_low)
        val_x_high = self.get_fpm_after_risk(val_x_high)
        '''
        Get alpha vales for F,P and M
        '''


        newdata_l = test_x_low[['F','P','M']]
        op_l = test_x_low['Difference']

        newdata_h = test_x_high[['F','P','M']]
        op_h = test_x_high['Difference']

        newdata_val_l = val_x_low[['F','P','M']]
        op_val_l = val_x_low['Difference']

        newdata_val_h = val_x_high[['F','P','M']]
        op_val_h = val_x_high['Difference']


        lr_low = LinearRegression().fit(newdata_l, op_l)	#Fitting and creating a model
        rfr_low = RandomForestRegressor().fit(newdata_l, op_l)

        lr_high = LinearRegression().fit(newdata_h, op_h)	#Fitting and creating a model
        rfr_high = RandomForestRegressor().fit(newdata_h, op_h)

        feature_importances_rfr_low = pd.DataFrame(rfr_low.feature_importances_,index = newdata_l.columns,columns=['importance']).sort_values('importance',ascending=False)
        feature_importance_lr_low = lr_low.coef_.T

        feature_importances_rfr_high = pd.DataFrame(rfr_high.feature_importances_,index = newdata_h.columns,columns=['importance']).sort_values('importance',ascending=False)
        feature_importance_lr_high = lr_high.coef_.T

        pred_lr_low = lr_low.predict(newdata_val_l)
        pred_lr_high = lr_high.predict(newdata_val_h)

        pred_rfr_low = rfr_low.predict(newdata_val_l)
        pred_rfr_high = rfr_high.predict(newdata_val_h)

        #print(mean_squared_error(pred_lr_low, op_val_l), r2_score(pred_lr_low, op_val_l))
        #print(mean_squared_error(pred_lr_high, op_val_h), r2_score(pred_lr_high, op_val_h))

        print(r2_model_1,r2_score_val, r2_score(pred_rfr_high, op_val_h))
        #print(feature_importances_rfr_low, feature_importance_lr_low)
        #print(feature_importance_lr_high, feature_importances_rfr_high)

        a_f = abs(feature_importance_lr_low[0])/(abs(feature_importance_lr_low[0])+abs(feature_importance_lr_low[1])+abs(feature_importance_lr_low[2]))
        a_p = abs(feature_importance_lr_low[1])/(abs(feature_importance_lr_low[0])+abs(feature_importance_lr_low[1])+abs(feature_importance_lr_low[2]))
        a_m = abs(feature_importance_lr_low[2])/(abs(feature_importance_lr_low[0])+abs(feature_importance_lr_low[1])+abs(feature_importance_lr_low[2]))
        b_f = abs(feature_importance_lr_high[0])/(abs(feature_importance_lr_high[0])+abs(feature_importance_lr_high[1])+abs(feature_importance_lr_high[2]))
        b_p = abs(feature_importance_lr_high[1])/(abs(feature_importance_lr_high[0])+abs(feature_importance_lr_high[1])+abs(feature_importance_lr_high[2]))
        b_m = abs(feature_importance_lr_high[2])/(abs(feature_importance_lr_high[0])+abs(feature_importance_lr_high[1])+abs(feature_importance_lr_high[2]))
        #feature_importances = pd.DataFrame(gbr.feature_importances_,index = train_x.columns,columns=['importance']).sort_values('importance',ascending=False)
        #print(feature_importances)
        #print("\n\nRisk=0:\nalpha-f: " + str(a_f) + "\nalpha-p: " + str(a_p) + "\nalpha-m: " + str(a_m) + '\n\n')
        #print("\n\nRisk=1:\nalpha-f: " + str(b_f) + "\nalpha-p: " + str(b_p) + "\nalpha-m: " + str(b_m) + '\n\n')
        return 1

    def create_model_with_ta(self):
        data = self.get_complete_data()
        data = data[np.isfinite(data['ROC'])]
        data = data[np.isfinite(data['AvgPosRet'])]
        data = data[np.isfinite(data['AvgNegRet'])]
        inp = data[['High','Open','Low','Close','Total Trade Quantity','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','AvgPosRet', 'AvgNegRet', 'MFI','RSI']]
        op = data[['Sn+1']]

        train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.20, random_state = 999)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.20, random_state=999)

        gbr = GradientBoostingRegressor().fit(train_x, train_y)	#Fitting and creating a model
        rfr = RandomForestRegressor().fit(train_x, train_y)

        joblib.dump(rfr, 'With_ta.pkl')

        feature_importances = pd.DataFrame(rfr.feature_importances_,index = train_x.columns,columns=['importance']).sort_values('importance',ascending=False)
        #print(feature_importances)
        return feature_importances


    def create_model_without_ta(self):
        data = self.get_complete_data()

        data = data[np.isfinite(data['ROC'])]
        inp = data[['High','Open','Low','Close','Total Trade Quantity','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','AvgPosRet', 'AvgNegRet']]
        op = data[['Sn+1']]

        train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.20, random_state = 999)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.20, random_state=999)
        gbr = GradientBoostingRegressor().fit(train_x, train_y)	#Fitting and creating a model
        rfr = RandomForestRegressor().fit(train_x, train_y)

        joblib.dump(rfr, 'Without_ta.pkl')
        feature_importances = pd.DataFrame(rfr.feature_importances_,index = train_x.columns,columns=['importance']).sort_values('importance',ascending=False)
        print(feature_importances)
        return feature_importances

    def predict_price(self, model):
        #self.create_model_with_ta()
        #self.create_model_without_ta()
        self.data = self.data[np.isfinite(data['ROC'])]
        self.data = self.data[np.isfinite(data['AvgPosRet'])]
        self.data = self.data[np.isfinite(data['AvgNegRet'])]
        actual = self.data['Sn+1']
        if(model == 'With_ta.pkl'):
            inp = self.data[['High','Open','Low','Close','Total Trade Quantity','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','AvgPosRet', 'AvgNegRet', 'MFI','RSI']]
        else:
            inp = self.data[['High','Open','Low','Close','Total Trade Quantity','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','AvgPosRet', 'AvgNegRet']]

        rfr_ta = joblib.load(model)
        #rfr_wta = joblib.load('Without_ta.pkl')
        pred_ta = rfr_ta.predict(inp)		#Predicting the answers for valdiation data
        #pred_wta = rfr_wta.predict(inp[['High','Open','Low','Close','Total Trade Quantity','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','AvgPosRet', 'AvgNegRet']])
        mse_ta = mean_squared_error(pred_ta, actual)	#finding the mean squared error
        #mse_wta = mean_squared_error(pred_wta, actual)	#finding the mean squared error
        '''
        rmse = math.sqrt(mse)
        for value in pred:
            value = value - rmse
        '''
        #print(len(actual), len(pred))
        score_ta = r2_score(pred_ta, actual)
        #score_wta = r2_score(pred_wta, actual)
        #print("\n\nR^2 Score (With TA) : " + str(score_ta) + "MSE (With TA) : " + str(mse_ta))
        #print("\n\nR^2 Score (Without TA) : " + str(score_wta) + "MSE (Without TA) : " + str(mse_wta) + '\n\n')
        self.data['S^n+1'] = pred_ta
        return pred_ta, score_ta, mse_ta, actual

    def get_fpm_values(self):
        F = []
        P = []
        M = []
        apr, anr = self.calculate_cummulative_avg_returns()
        self.predict_price('With_ta.pkl')
        close_n = self.data['Close'].tolist()
        close_n1 = self.data['S^n+1'].tolist()

        #print(len(close_n), len(close_n1))
        for i in range(0,len(close_n)):
            if(close_n1[i]-close_n[i] > 0):
                F += [close_n1[i]-close_n[i]]
                P += [apr*close_n[i]]
                M += [anr*close_n[i]]
            else:
                F += [close_n1[i]-close_n[i]]
                P += [anr*close_n[i]]
                M += [apr*close_n[i]]
        self.data['Full'] = F
        self.data['Partial'] = P
        self.data['Mis'] = M

        #print(self.data.columns)
        return self.data

    def calculate_constants(self):
        self.get_fpm_values()
        inp = self.data[['Full','Partial','Mis']]
        op = self.data['Difference']
        train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.25, random_state = 999)
        lr = LinearRegression().fit(train_x, train_y)	#Fitting and creating a model
        rfr = RandomForestRegressor().fit(train_x, train_y)
        feature_importances = pd.DataFrame(rfr.feature_importances_,index = train_x.columns,columns=['importance']).sort_values('importance',ascending=False)

        feature_importance_lr = lr.coef_.T

        print(feature_importances, feature_importance_lr)

if(__name__ == '__main__'):
    data = pd.read_csv("datasets/HistoricData_NTPC.csv")
    #self.data = add_all_ta_features(self.data,"Open","High","Low","Close","Total Trade Quantity", fillna=True)
    ta_obj = TechnicalAnalysis(data)
    data = ta_obj.get_complete_data()
    importances = ta_obj.create_model_with_ta()

    ta_obj.calculate_constants()

    ta_obj.ta_pred_model(data)
    #print(data.columns)
