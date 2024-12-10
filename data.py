import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler,LabelEncoder,RobustScaler

"""
prepare_datasets() -> train데이터프레임, test데이터프레임
get_Catcol_nunique() -> list    categorical unique값 계산
"""



CAT_COLS=['AB_DM','AB_HTN','AN_ASA','AN_Asthma','AN_COPD',
'AN_Heart_Dz','AN_Hematologic_Dz','AN_Liver_Dz','AN_Neurologic_Dz','AN_NYHA',
'AN_Other_organ_Dz','AN_Pregnancy','AN_Renal_Dz','AN_TB','AN_Thyroid_Dz',
'AN_Vascular_Dz','A_SMK2','A_Sex','B_AKI','B_CAD','B_CKD','B_COPD',
'B_CVD','B_Malig','B_UALB','B_URBC','D_Aspirin_14',
'D_Aspirin_90','D_Clopidogrel_14','D_Clopidogrel_90','D_DIURETICS14','D_DIURETICS90',
'D_Ezetimibe_14','D_Ezetimibe_90','D_Fenofibrate_14','D_Fenofibrate_90','D_ISA_14',
'D_ISA_90','D_LMWH_14','D_LMWH_90','D_NOAC_14','D_NOAC_90',
'D_NSAID14','D_NSAID90','D_RASB_14','D_RASB_90','D_Statin_14',
'D_Statin_90','D_Steroid_14','D_Steroid_90','D_Warfarin_14','D_Warfarin_90',
'Op_AN','Op_Dep','Op_Type','Type_Adm','PP','A_DBP_cut','A_SBP_cut','A_BMI']

NUM_COLS=['A_Age','A_HR','A_HT','A_WT','B_Alb',
'B_ALP','B_ALT','B_AST','B_BIL','B_BUN',
'B_Ca','B_Chol','B_CL','B_Cr_near','B_ESR',
'B_Glucose','B_Hb','B_HbA1c','B_Hct','B_hsCRP',
'B_INR','B_K','B_Na','B_Neutrophil','B_P',
'B_Plt','B_Protein','B_tCO2','B_Uric','B_WBC',
'Dur_Adm_before_op','Op_EST_Dur']


TARGETS=[
'O_AKI','O_Critical_AKI_90',
'O_Death_90','O_RRT_90','Synthetic_type']




def Feature_engineering(dataframe):
    """데이터 프리프로세싱 및 특성 엔지니어링
        -이완기 및 수축기 혈압 범주화
        -BMI 컬럼 생성 및 범주화 진행
        - A_DBP : 이완기 혈압(수치형) -> A_DBP_cut(범주)
        - A_SBP: 수축기 혈압(수치형) -> A_SBP_cut(범주)
        - PP: SBP-DBO (맥압)(수치형) -> PP(범주)
    Args:
        dataframe (pandas.DataFrame): 변환될 데이터프레임
    """
    idx=dataframe[dataframe['A_SBP']<dataframe['A_DBP']].index  
    dataframe.loc[idx,'A_SBP']=np.nan
    dataframe.loc[idx,'A_DBP']=np.nan # 수축기 보다 이완기가 높은 경우 제거
    
    dataframe['PP']=pd.cut((dataframe['A_SBP']-dataframe['A_DBP']),bins=[0,40,60,200],labels=['저맥압','정상','고맥압'])
    dataframe['A_DBP_cut']=pd.cut(dataframe.A_DBP,bins=[0,80,90,160],labels=['저혈압','정상','고혈압'])
    dataframe['A_SBP_cut']=pd.cut(dataframe.A_SBP,bins=[0,120,140,250],labels=['저혈압','정상','고혈압']) 
    
    dataframe[dataframe['A_Sex']==1]['A_WT']=dataframe[dataframe['A_Sex']==1]['A_WT'].fillna(dataframe[dataframe['A_Sex']==1]['A_WT'].median())
    dataframe[dataframe['A_Sex']==2]['A_WT']=dataframe[dataframe['A_Sex']==2]['A_WT'].fillna(dataframe[dataframe['A_Sex']==2]['A_WT'].median())
    dataframe[dataframe['A_Sex']==1]['A_HT']=dataframe[dataframe['A_Sex']==1]['A_HT'].fillna(dataframe[dataframe['A_Sex']==1]['A_HT'].median())
    dataframe[dataframe['A_Sex']==2]['A_HT']=dataframe[dataframe['A_Sex']==2]['A_HT'].fillna(dataframe[dataframe['A_Sex']==2]['A_HT'].median())
    dataframe['A_BMI']=pd.cut(dataframe['A_WT']/((dataframe['A_HT']/100)**2),bins=[0,18.5,25,30,110],labels=[0,1,2,3])  # 
    
    dataframe.drop(columns= ['A_DBP','A_SBP'],inplace=True)
    dataframe.drop(columns= ['B_UPCR','B_Triglyceride','B_PTH','B_HDL','B_LDL'],inplace=True)  # 결측치 많은 컬럼 제거 
    
    print("Feature Engineering 완료")
    if set(TARGETS).issubset(set(dataframe.columns)):
        return dataframe[CAT_COLS+NUM_COLS+TARGETS] 
    else:
        return dataframe[CAT_COLS+NUM_COLS]
        


# Synthetic type기준으로 splitting
def split_validation(dataframe,value=0):
    valid_dataframe=dataframe[dataframe['Synthetic_type']==value]
    train_dataframe = dataframe[dataframe['Synthetic_type']!=value]
    return train_dataframe,valid_dataframe

def preprocessing(train_dataframe,test_dataframe,num_cols,cat_cols):
    s=RobustScaler()
    le=LabelEncoder()

    train_dataframe[num_cols]=s.fit_transform(train_dataframe[num_cols])
    train_dataframe[num_cols]=train_dataframe[num_cols].fillna(0)
    test_dataframe[num_cols]=s.transform(test_dataframe[num_cols])
    test_dataframe[num_cols]=test_dataframe[num_cols].fillna(0)

    for i in cat_cols:
        train_dataframe[i]=le.fit_transform(train_dataframe[i])
        test_dataframe[i]=le.transform(test_dataframe[i])

    print("Preprocessing 완료")
    return train_dataframe,test_dataframe

    
def prepare_datasets():
    train_df = pd.read_csv('Train_SyntheticAKI_MAIC2023.csv')
    test_df = pd.read_csv('Test_SyntheticAKI_MAIC2023.csv')
    columns =pd.read_csv("columns.csv")
    num_cols = NUM_COLS
    cat_cols = CAT_COLS
    train_df = Feature_engineering(train_df)
    test_df = Feature_engineering(test_df)
    train_df,test_df = preprocessing(train_df,test_df,num_cols,cat_cols)
    return train_df,test_df

def get_Catcol_nunique(dataframe):
    cats=[]
    for i in CAT_COLS:
        cats.append(dataframe[i].nunique())
    return cats
    

