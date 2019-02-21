import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

churn_df = pd.read_csv('D:\Project_work\Data\Training_Dataset_1.csv')
col_names = churn_df.columns.tolist()
     
# We don't need these columns
to_drop = ['City','Gender','Location Code','Marital Status','Effective To Date']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
churn_feat_space['Response'] = np.where(churn_feat_space.Response.str.contains('Yes') == True, 1, churn_feat_space.Response)
churn_feat_space['Response'] = np.where(churn_feat_space.Response.str.contains('No') == True, 0, churn_feat_space.Response)

churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Extended') == True, 3, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Premium') == True, 2, churn_feat_space.Coverage)
churn_feat_space['Coverage'] = np.where(churn_feat_space.Coverage.str.contains('Basic') == True, 1, churn_feat_space.Coverage)
#print(churn_feat_space['Coverage'])

churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Grade X') == True, 1, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Grade XII') == True, 2, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('PhD') == True, 6, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Diploma') == True, 3, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Graduate') == True, 4, churn_feat_space.Education)
churn_feat_space['Education'] = np.where(churn_feat_space.Education.str.contains('Master') == True, 5, churn_feat_space.Education)


churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('Unemployed') == True, 0, churn_feat_space.Employment_Status)
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('Employed') == True, 3, churn_feat_space.Employment_Status)
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('On leave') == True, 1, churn_feat_space.Employment_Status)
churn_feat_space['Employment_Status'] = np.where(churn_feat_space.Employment_Status.str.contains('Retired') == True, 2, churn_feat_space.Employment_Status)
#print(churn_feat_space['Employment_Status'])

churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Corporate') == True, 4, churn_feat_space.Policy_Type)
churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Special') == True, 3, churn_feat_space.Policy_Type)
churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Privileged') == True, 2, churn_feat_space.Policy_Type)
churn_feat_space['Policy_Type'] = np.where(churn_feat_space.Policy_Type.str.contains('Personal') == True, 1, churn_feat_space.Policy_Type)
#print(churn_feat_space['Policy_Type'])

churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Bad') == True, 1, churn_feat_space.Policy_Rating)
churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Good') == True, 3, churn_feat_space.Policy_Rating)
churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Average') == True, 2, churn_feat_space.Policy_Rating)
churn_feat_space['Policy_Rating'] = np.where(churn_feat_space.Policy_Rating.str.contains('Excellent') == True, 4, churn_feat_space.Policy_Rating)

churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 1') == True, 1, churn_feat_space.Renew_Offer_Type)
churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 2') == True, 2, churn_feat_space.Renew_Offer_Type)
churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 3') == True, 3, churn_feat_space.Renew_Offer_Type)
churn_feat_space['Renew_Offer_Type'] = np.where(churn_feat_space.Renew_Offer_Type.str.contains('Offer 4') == True, 4, churn_feat_space.Renew_Offer_Type)


churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Online') == True, 2, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Branch') == True, 4, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Call Center') == True, 1, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Agent') == True, 5, churn_feat_space.Sales_Channel)
churn_feat_space['Sales_Channel'] = np.where(churn_feat_space.Sales_Channel.str.contains('Other') == True, 3, churn_feat_space.Sales_Channel)

churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Excellent') == True, 6, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Good') == True, 5, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Satisfactory') == True, 4, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Average') == True, 3, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Bad') == True, 2, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Horrible') == True, 1, churn_feat_space.Feedback)
churn_feat_space['Feedback'] = np.where(churn_feat_space.Feedback.str.contains('Not Given') == True, 0, churn_feat_space.Feedback)
#print(churn_feat_space)

#churn_feat_space.to_csv('D:\Project_work\Data\out.csv')
# Function to get data
def get_data(file_name):
    data= pd.read_csv(file_name)
    customer_a_parameter = []
    customer_b_parameter = []
    customer_c_parameter = []
    customer_d_parameter = []
    customer_e_parameter = []
    customer_f_parameter = []
    customer_g_parameter = []
    customer_h_parameter = []
    customer_i_parameter = []
    customer_j_parameter = []
    customer_k_parameter = []
    customer_l_parameter = []
    customer_m_parameter = []
    customer_n_parameter = []
    customer_o_parameter = []
    customer_p_parameter = []
    customer_q_parameter = []
    customer_r_parameter = []
    customer_s_parameter = []
    customer_t_parameter = []
    customer_u_parameter = []
    customer_v_parameter = []
    customer_w_parameter = []
    customer_x_parameter = []
    customer_y_parameter = []




    
    for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y in zip(data['Customer ID'],
                                                             data['Customer Lifetime Value'],
                                                             data['Response'],
                                                             data['Coverage'],
                                                             data['Education'],
                                                             data['Employment_Status'],
                                                             data['Income'],
                                                             data['Location Code'],
                                                             data['Monthly Premium Auto'],
                                                             data['Total Written Premium'],
                                                             data['Losses'],
                                                             data['Loss Ratio'],
                                                             data['Growth Rate'],
                                                             data['Commissions'],
                                                             data['Months Since Last Claim'],
                                                             data['Months Since Policy Inception'],
                                                             data['Number of Open Complaints'],
                                                             data['Number of Policies'],
                                                             data['Number of previous policies'],
                                                             data['Policy_Type'],
                                                             data['Policy_Rating'],
                                                             data['Renew_Offer_Type'],
                                                             data['Sales_Channel'],
                                                             data['Total Claim Amount'],
                                                             data['Feedback']):
        customer_a_parameter.append([float(a)])
        customer_b_parameter.append([float(b)])
        customer_c_parameter.append([float(c)])
        customer_d_parameter.append([float(d)])
        customer_e_parameter.append([float(e)])
        customer_f_parameter.append([float(f)])
        customer_g_parameter.append([float(g)])
        customer_h_parameter.append([float(h)])
        customer_i_parameter.append([float(i)])
        customer_j_parameter.append([float(j)])
        customer_k_parameter.append([float(k)])
        customer_l_parameter.append([float(l)])
        customer_m_parameter.append([float(m)])
        customer_n_parameter.append([float(n)])
        customer_o_parameter.append([float(o)])
        customer_p_parameter.append([float(p)])
        customer_q_parameter.append([float(q)])
        customer_r_parameter.append([float(r)])
        customer_s_parameter.append([float(s)])
        customer_t_parameter.append([float(t)])
        customer_u_parameter.append([float(u)])
        customer_v_parameter.append([float(v)])
        customer_w_parameter.append([float(w)])
        customer_x_parameter.append([float(x)])
        customer_y_parameter.append([float(y)])

    return customer_a_parameter,customer_b_parameter,customer_c_parameter, customer_d_parameter,customer_e_parameter,customer_f_parameter,customer_g_parameter,customer_h_parameter,customer_i_parameter,customer_j_parameter,customer_k_parameter,customer_l_parameter,customer_m_parameter,customer_n_parameter,customer_o_parameter,customer_q_parameter,customer_r_parameter,customer_s_parameter,customer_t_parameter,customer_u_parameter,customer_v_parameter,customer_w_parameter,customer_x_parameter,customer_y_parameter
 
# Function to know which Tv show will have more viewers
def more_viewers(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y):
    regr1 = linear_model.LinearRegression()
    regr1.fit(a, b)
    predicted_value1 = regr1.predict(15)
    print (predicted_value1)
    
    regr2 = linear_model.LinearRegression()
    regr2.fit(a, c)
    predicted_value2 = regr2.predict(15)
    print (predicted_value2)
    
    regr3 = linear_model.LinearRegression()
    regr3.fit(a, d)
    predicted_value3 = regr3.predict(15)
    print (predicted_value3)
    
    regr4 = linear_model.LinearRegression()
    regr4.fit(a, e)
    predicted_value4 = regr4.predict(15)
    print (predicted_value4)
    
    regr5 = linear_model.LinearRegression()
    regr5.fit(a, f)
    predicted_value5 = regr5.predict(15)
    print (predicted_value5)
    
    regr6 = linear_model.LinearRegression()
    regr6.fit(a, g)
    predicted_value6 = regr6.predict(15)
    print (predicted_value6)
    
    regr7 = linear_model.LinearRegression()
    regr7.fit(a, h)
    predicted_value7 = regr7.predict(15)
    print (predicted_value7)
    
    regr8 = linear_model.LinearRegression()
    regr8.fit(a, i)
    predicted_value8 = regr8.predict(15)
    print (predicted_value8)
    
    regr9 = linear_model.LinearRegression()
    regr9.fit(a, j)
    predicted_value9 = regr9.predict(15)
    print (predicted_value9)
    
    regr10 = linear_model.LinearRegression()
    regr10.fit(a, k)
    predicted_value10 = regr10.predict(15)
    print (predicted_value10)
    
    regr11 = linear_model.LinearRegression()
    regr11.fit(a, l)
    predicted_value11 = regr11.predict(15)
    print (predicted_value11)
    
    regr12 = linear_model.LinearRegression()
    regr12.fit(a, m)
    predicted_value12 = regr12.predict(15)
    print (predicted_value12)
    
    regr13 = linear_model.LinearRegression()
    regr13.fit(a, n)
    predicted_value13 = regr13.predict(15)
    print (predicted_value13)
    
    regr14 = linear_model.LinearRegression()
    regr14.fit(a, o)
    predicted_value14 = regr14.predict(15)
    print (predicted_value14)
    
    regr15 = linear_model.LinearRegression()
    regr15.fit(a, p)
    predicted_value15 = regr15.predict(15)
    print (predicted_value15)
    
    regr16 = linear_model.LinearRegression()
    regr16.fit(a, q)
    predicted_value16 = regr16.predict(15)
    print (predicted_value16)
    
    regr17 = linear_model.LinearRegression()
    regr17.fit(a, r)
    predicted_value17 = regr17.predict(15)
    print (predicted_value17)
    
    regr18 = linear_model.LinearRegression()
    regr18.fit(a, s)
    predicted_value18 = regr18.predict(15)
    print (predicted_value18)
    
    regr19 = linear_model.LinearRegression()
    regr19.fit(a, t)
    predicted_value19 = regr19.predict(15)
    print (predicted_value19)
    
    regr20 = linear_model.LinearRegression()
    regr20.fit(a, u)
    predicted_value20 = regr20.predict(15)
    print (predicted_value20)
    
    regr21 = linear_model.LinearRegression()
    regr21.fit(a, v)
    predicted_value21 = regr21.predict(15)
    print (predicted_value21)
    
    regr22 = linear_model.LinearRegression()
    regr22.fit(a, w)
    predicted_value22 = regr22.predict(15)
    print (predicted_value22)
    
    regr23 = linear_model.LinearRegression()
    regr23.fit(a, x)
    predicted_value23 = regr23.predict(15)
    print (predicted_value23)
    
    regr24 = linear_model.LinearRegression()
    regr24.fit(a, y)
    predicted_value24 = regr24.predict(15)
    print (predicted_value24)
    
   
    pred1=0 
    pred2=0 
    pred3=0 
    pred4=0 
    pred5=0 
    pred6=0 
    pred7=0 
    pred8=0 
    pred9=0 
    pred10=0 
    pred11=0 
    pred12=0 
    pred13=0 
    pred14=0 
    pred15=0 
    pred16=0 
    pred17=0 
    pred18=0 
    pred19=0 
    pred20=0 
    pred21=0 
    pred22=0 
    pred23=0 
    pred24=0 

 
    if(predicted_value1>10):
        pred1+=1
    if(predicted_value2>10):
        pred2+=1
    if(predicted_value3>10):
        pred3+=1
    if(predicted_value4>10):
        pred4+=1
    if(predicted_value5>10):
        pred5+=1
    if(predicted_value6>10):
        pred6+=1
    if(predicted_value7>10): 
        pred7+=1
    if(predicted_value8>10): 
        pred8+=1
    if(predicted_value9>10): 
        pred9+=1
    if(predicted_value10>10): 
        pred10+=1
    if(predicted_value11>10):
        pred11+=1
    if(predicted_value12>10):
        pred12+=1
    if(predicted_value13>10):
        pred13+=1
    if(predicted_value14>10):
        pred14+=1
    if(predicted_value15>10):
        pred15+=1
    if(predicted_value16>10):
        pred16+=1
    if(predicted_value17>10): 
        pred17+=1
    if(predicted_value18>10): 
        pred18+=1
    if(predicted_value19>10): 
        pred19+=1
    if(predicted_value20>10): 
        pred20+=1
    if(predicted_value21>10): 
        pred21+=1
    if(predicted_value22>10): 
        pred22+=1
    if(predicted_value23>10): 
        pred23+=1
    if(predicted_value24>10): 
        pred24+=1
        
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
        
    if pred2==0:
        print("Customer 2 will remain")
    if(pred2==2):
        print("Customer 2 will leave")
                
    if pred2==0:
        print("Customer 2 will remain")
    if(pred2==2):
        print("Customer 2 will leave")
                
    if pred2==0:
        print("Customer 2 will remain")
    if(pred2==2):
        print("Customer 2 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
    if pred1==0:
        print("Customer 1 will remain")
    if(pred1==1):
        print("Customer 1 will leave")
                
        
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y = get_data(churn_feat_space)
# Call the function more_viewers to predict the more viewers television show
more_viewers(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y)
