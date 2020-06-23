#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:06:50 2020

@author: pranayrastogi
"""
import os,sys
import pandas as pd
import numpy as np
from configparser import ConfigParser
from sklearn.ensemble import RandomForestClassifier

class UtilityFuncs():
     def getAgeRange(age):
        if age < 27: return 1
        elif age < 40: return 2
        elif age < 50: return 3
        elif age < 65: return 4
        elif age < 99: return 5
        else: return 0

class HomeCredit():
    def __init__(self,config):
        self.iniconfig = config
        self.loanData = self.treatApplData()    
        self.main()
   
    def dropAppDataCols(self,df):
    
        drop_list = [
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
            'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
            'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
            'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
            'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
            'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
            'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
            'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
            'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 
            'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
            'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
            'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
            'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG']
        
   
        for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
            drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
        df.drop(drop_list, axis=1, inplace=True)
        return df

    def treatApplData(self):
        application_train = pd.read_csv(self.iniconfig.get('MAIN',"APPLICATION_TRAIN_DATA"))
        application_test = pd.read_csv(self.iniconfig.get('MAIN',"APPLICATION_TEST_DATA"))
        df = application_train.append(application_test)
        df = df[df['CODE_GENDER'] != 'XNA'] 
        df = df[df['AMT_INCOME_TOTAL'] < df['AMT_INCOME_TOTAL'].quantile(0.9995)]
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        docs = [f for f in df.columns if 'FLAG_DOC' in f]
        df['TOTAL_DOCUMENTS'] = df[docs].sum(axis=1)
        df['YEARS_BIRTH'] = df['DAYS_BIRTH'] / (-365)
        df['YEARS_BIRTH'] = df['YEARS_BIRTH'].apply(lambda x:UtilityFuncs.getAgeRange(x))
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])

        df['ANNUITY_PER_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']    
        object_columns = [col for col in df.columns if df[col].dtype == 'object']
        original_columns = list(df.columns)
        df = pd.get_dummies(df, columns= object_columns, dummy_na= False)
        df = self.dropAppDataCols(df)
        return df
    
    def posCashData(self):
        poscash = pd.read_csv(self.iniconfig.get('MAIN',"POS_CASH_DATA"))
        cat_cols = [col for col in poscash.columns if poscash[col].dtype=="object"]

        poscash = pd.get_dummies(poscash,columns = cat_cols,dummy_na= True)

        aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
        }

        pos_agg = poscash.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        pos_agg['POS_COUNT'] = poscash.groupby('SK_ID_CURR').size()        
        pos_agg=pd.merge(poscash,pos_agg,how="left",on="SK_ID_CURR")
        pos_agg.drop(["MONTHS_BALANCE","SK_ID_PREV"],axis=1,inplace=True)
        return pos_agg
        
    def installmentData(self):
        installments = pd.read_csv(self.iniconfig.get('MAIN',"INSTALLMENT_DATA"))
        installments['PAYMENT_RATIO'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
        installments['PAY_DIFFERENCE'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
        installments['DAYS_PAST_DUE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
        installments['DAYS_BEFORE_DUE'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
        installments['DAYS_PAST_DUE'] = installments['DAYS_PAST_DUE'].apply(lambda x: x if x > 0 else 0)
        installments['DAYS_BEFORE_DUE'] = installments['DAYS_BEFORE_DUE'].apply(lambda x: x if x > 0 else 0)
        inst = installments.groupby('SK_ID_CURR').agg({'DAYS_PAST_DUE':'mean','DAYS_BEFORE_DUE':'mean','PAY_DIFFERENCE':['max','min','mean','sum'],'PAYMENT_RATIO':['max','min','mean','sum']})
        inst.columns = pd.Index(['Installment_' + col[0].upper() + "_" + col[1].upper() for col in inst.columns.tolist()])
        inst['INSTAL_COUNT'] = installments.groupby('SK_ID_CURR').size()
        return inst
        
        
    def prevAppData(self):
        previousapp =  pd.read_csv(self.iniconfig.get('MAIN',"PREV_APP"))
        object_col = [col for col in previousapp.columns if previousapp[col].dtype=="object"]
        previousapp = pd.get_dummies(previousapp,columns=object_col,dummy_na=False)
        previousapp['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        previousapp['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        previousapp['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        previousapp['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        previousapp['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)        
        previousapp['APPLICATION_CREDIT_DIFF'] = previousapp['AMT_APPLICATION'] - previousapp['AMT_CREDIT']
        previousapp['APPLICATION_CREDIT_RATIO'] = previousapp['AMT_APPLICATION'] / previousapp['AMT_CREDIT']
        previousapp['CREDIT_TO_ANNUITY_RATIO'] = previousapp['AMT_CREDIT']/previousapp['AMT_ANNUITY']
        previousapp['DOWN_PAYMENT_TO_CREDIT'] = previousapp['AMT_DOWN_PAYMENT'] / previousapp['AMT_CREDIT']
        previousapp.drop('SK_ID_PREV',axis=1, inplace=True)
        return previousapp
     
        
    def bureauData(self):
        bureau = pd.read_csv(self.iniconfig.get('MAIN',"BUREAU_DATA"))
        object_col = [col for col in bureau.columns if bureau[col].dtype=="object"]
        bureau = pd.get_dummies(bureau,columns=object_col,dummy_na=False)
        bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
        bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
        bureau = pd.merge(bureau,self.bureauBalData(),how='left',on="SK_ID_BUREAU")
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        return bureau
       
        
    def bureauBalData(self):
        bureaubal = pd.read_csv(self.iniconfig.get('MAIN',"BUREAU_BAL"))
        object_col = [col for col in bureaubal.columns if bureaubal[col].dtype=="object"]
        bureaubal = pd.get_dummies(bureaubal,columns=object_col,dummy_na=False)
        bb_agg = bureaubal.groupby("SK_ID_BUREAU").agg({'MONTHS_BALANCE':['mean','sum']})
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bb_agg = pd.merge(bureaubal,bb_agg,on="SK_ID_BUREAU",how="left")
        bb_agg.drop("MONTHS_BALANCE",axis=1,inplace=True)
        return bb_agg
        
        
    def creditCardBalance(self):
        ccbalance =  pd.read_csv(self.iniconfig.get('MAIN',"CC_BALANCE"))
        object_columns = [col for col in ccbalance.columns if ccbalance[col].dtype=="object"]
        ccbalance = pd.get_dummies(ccbalance, columns= object_columns, dummy_na= True)
        ccbalance['LIMIT_USE'] = ccbalance['AMT_BALANCE'] / ccbalance['AMT_CREDIT_LIMIT_ACTUAL']  
        ccbalance['DELAYED_PAYMENT'] = ccbalance['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
        ccbalance.drop(['SK_ID_PREV'], axis= 1, inplace = True)
        ccdata = ccbalance.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum'])
        ccdata.columns = pd.Index(['CredCard_' + col[0].upper() + "_" + col[1].upper() for col in ccdata.columns.tolist()])
        ccdata['CredCard_COUNT'] = ccbalance.groupby('SK_ID_CURR').size()
        return ccdata
     
        
    def fitmodel(self,num_folds=10):
        train_df = self.loanData[self.loanData['TARGET'].notnull()]
        test_df = self.loanData[self.loanData['TARGET'].isnull()]
        train_labels = train_df.TARGET
        train_df.drop("TARGET",axis=1,inplace=True)
        random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
        random_forest.fit(train_df, train_labels)
        predictions = random_forest.predict_proba(test_df)[:, 1]
        submit = test_df[['SK_ID_CURR']]
        submit['TARGET'] = predictions
        submit.to_csv('homecredit_result.csv', index = False)
        
         
    def main(self):
        self.loanData = pd.merge(self.loanData ,self.posCashData(),how="left",on="SK_ID_CURR")   
        self.loanData = pd.merge(self.loanData ,self.installmentData() ,how="left",on="SK_ID_CURR")  
        self.loanData = pd.merge(self.loanData ,self.prevAppData(),how="left",on="SK_ID_CURR") 
        self.loanData = pd.merge(self.loanData ,self.bureauData()  ,how="left",on="SK_ID_CURR")
        self.loanData = pd.merge(self.loanData ,self.creditCardBalance() ,how="left",on="SK_ID_CURR")         
        self.fitmodel(num_folds=5)
        
if __name__ == "__main__":
    config = ConfigParser()
    config.read(sys.argv[1])
    obj = HomeCredit(config)
    
