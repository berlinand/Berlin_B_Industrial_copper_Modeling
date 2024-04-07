
from pymongo import MongoClient
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static

#mg_connection will connect monodb cluster to access databases and collections
mg_connection=MongoClient("mongodb://localhost:27017")

mgdb=mg_connection["sample_airbnb"]
mgcol=mgdb["listingsAndReviews"]

#This function convert mongodb datas into pandas dataframe
def convert_mgcol_pd():
    datas=[]
    for x in mgcol.find():
      datas.append(x)
    df=pd.DataFrame(datas)
    df['bathrooms']=df['bathrooms'].apply(str).astype(float)
    df['price']=df['price'].apply(str).astype(float)
    df['security_deposit']=df['security_deposit'].apply(str).astype(float)
    df['cleaning_fee']=df['cleaning_fee'].apply(str).astype(float)
    df['extra_people']=df['extra_people'].apply(str).astype(float)
    df['guests_included']=df['guests_included'].apply(str).astype(int)
    df['weekly_price']=df['weekly_price'].apply(str).astype(float)
    df['monthly_price']=df['monthly_price'].apply(str).astype(float)
    df['reviews']=df['reviews'].apply(tuple)
    df['images']=df['images'].apply(tuple)
    df['host']=df['host'].apply(tuple)
    df['address']=df['address'].apply(tuple)
    df['availability']=df['availability'].apply(tuple)
    df['review_scores']=df['review_scores'].apply(tuple)
    df['reviews']=df['reviews'].apply(lambda x:tuple(str(s) for s in x))
    df['reviews']=df['reviews'].apply(lambda x:tuple(tuple(s) for s in x))
    df['amenities']=df['amenities'].apply(tuple)
    df['address']=df['address'].apply(str)
    address=[]
    for y in range(0,len(datas)):
       row=datas[y]['address']
       address.append(row)
    df1=pd.DataFrame(address)
    coordinates=[]
    for z in range(0,len(datas)):
       row=datas[z]['address']['location']['coordinates'] 
       coordinates.append(row)
    df2=pd.DataFrame(coordinates,columns=['longitude','latitude'])

    reviews_score=[]
    for a in range(0,len(datas)):
       row=datas[a]["review_scores"]
       reviews_score.append(row)
    df3=pd.DataFrame(reviews_score,columns=['review_scores_accuracy',
                                          'review_scores_cleanliness',
                                          'review_scores_checkin',
                                          'review_scores_communication',
                                          'review_scores_location',
                                          'review_scores_value',
                                          'review_scores_rating'
                                          ] )
    avaliablity=[]
    for b in range(0,len(datas)):
       row=datas[b]['availability']
       avaliablity.append(row)
    df4=pd.DataFrame(avaliablity,columns=['availability_30',
                                          'availability_60',
                                          'availability_90',
                                          'availability_365'
                                          ] )

    df=pd.concat([df,df1,df2,df3,df4],axis=1)

    return df
#This function analyse and drop unwanted columns in dataframe
def droping(df):
      df=df.drop('summary',axis=1)
      df=df.drop('space',axis=1)
      df=df.drop('neighborhood_overview',axis=1)
      df=df.drop('notes',axis=1)
      df=df.drop('transit',axis=1)
      df=df.drop('first_review',axis=1)
      df=df.drop('last_review',axis=1)
      df=df.drop('reviews_per_month',axis=1)
      df=df.drop('address',axis=1)
      df=df.drop('location',axis=1)
      df=df.dropna(subset=['availability_30','availability_60','availability_90','availability_365'])

      return df

#This function fill up the  nan values and delete small quantity of rows having nan value
def del_add_values(df):
    df=df[df['name']!='']
    x=df['name'].duplicated()
    df=df[x==False]
    df['beds']=df['beds'].fillna(0.0)
    df['bedrooms']=df['bedrooms'].fillna(0.0)
    df['bathrooms']=df['bathrooms'].fillna(1.0)
    df['weekly_price']=df['weekly_price'].fillna(7*df['price'])
    df['monthly_price']=df['monthly_price'].fillna(30*df['price'])
    df['cleaning_fee']=df['cleaning_fee'].fillna(df['cleaning_fee'].mean())
    df['security_deposit']=df['security_deposit'].fillna(df['security_deposit'].mean())
    df['name'] = df['name'].replace('{{ SAGRADA FAMILIA }} Center ROOM *****', 'SAGRADA FAMILIA- Center ROOM')
    df=df.drop(["review_scores"],axis=1)
    df['review_scores_accuracy']=df['review_scores_accuracy'].fillna(df['review_scores_accuracy'].mean())
    df['review_scores_cleanliness']=df['review_scores_cleanliness'].fillna(df['review_scores_cleanliness'].mean())
    df['review_scores_rating']=df['review_scores_rating'].fillna(df['review_scores_rating'].mean())
    df['review_scores_value']=df['review_scores_value'].fillna(df['review_scores_value'].mean())

    return df
#This function  display a world map along with location of property in different country  and some details of property 
def map(df,dp1,dp2a,dp2,dp3,op2):
      try:

          if 'select all country' in dp1:
             df_li=list(df['country'].unique())

             df1=df[df['country'].isin(df_li)].head(1000)

          else:
            df1=df[df['country'].isin(dp1)]
          if dp2a=="less than":
            if dp2!="select all":
               df1=df1[df1['price']<dp2]
            else:
               df1=df1
          else:
            if dp2!="select all":
              df1=df1[df1['price']>dp2]
            else:
               df1=df1
            
          if dp2a=="less than":
            if dp2!="select all":
               df1=df1[df1['weekly_price']<dp2]
            else:
               df1=df1
          else:
            if dp2!="select all":
              df1=df1[df1['weekly_price']>dp2]
            else:
               df1=df1
          if dp2a=="less than":
            if dp2!="select all":
               df1=df1[df1['monthly_price']<dp2]
            else:
               df1=df1
          else:
            if dp2!="select all":
              df1=df1[df1['monthly_price']>dp2]
            else:
               df1=df1
          if dp2a=="less than":
            if dp3!="select all":
              df1=df1[df1['bedrooms']<dp3]
            else:
               df1=df1
          else:
            if dp3!="select all":
                df1=df1[df1['bedrooms']>dp3]
            else:
               df1=df1
          if "select all" in op2:
             df1=df1
          else:
             df1=df1[df1['property_type']].isin(op2)
          loc_center = [df1['latitude'].mean(), df1['longitude'].mean()]
          map1 = folium.Map(location = loc_center, zoom_start = 0, control_scale=True)
          for index, loc in df1.iterrows():
              Tooltip =f"""Name: {loc['name']}<br>property type: {loc['property_type']}<br>room type: {loc['room_type']}
                                         <br>Bedroom: {loc['bedrooms']}<br>Street: {loc['street']}
                       <br>Country: {loc['country']}<br>No of review: {loc['number_of_reviews']}<br>Rating: {loc['review_scores_rating']}
                     <br>Daily price: {loc['price']}<br>weekly price: {loc['weekly_price']} <br>monthly price: {loc['monthly_price']} 
                    <br>security deposit: {loc['security_deposit']}<br>cleaning fees: {loc['cleaning_fee']}
                    <br>minimum_nights: {loc['minimum_nights']}<br>maximum_nights : {loc['maximum_nights']}
                 """
              folium.CircleMarker([loc['latitude'], loc['longitude']], tooltip=Tooltip, radius=2, weight=5, popup=loc['name']).add_to(map1)
          folium.LayerControl().add_to(map1) 
      except:
          st.warning("In this country there are no more higher value ,please select lesser value")       
          
      return map1,df1  

#This function display different types of chart for Visualization of price
def price_charts(df1,chart_type,X):
   
    if chart_type=='scatter chart':
        if X=="season":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[a[0]*df1['price'].sum(),a[1]*df1['price'].sum(),a[2]*df1['price'].sum(),a[3]*df1['price'].sum()]
          X=['30 days','60 days','90 days','365 days']
        elif X=="Name":
           X='name'
           Y='price'
        elif X=='property types':
          X='property_type'
          Y='price'
        elif X=='country':
          X='country'
          Y='price'
        fig1=px.scatter(
          df1,
          x=X,
          y=Y,
          title="Scatter plots chart"
           )
       
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 
    if chart_type=='pie chart':
        if X=="season":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[(a[3]-a[0])*df1['price'].sum(),(a[3]-a[1])*df1['price'].sum(),(a[3]-a[2])*df1['price'].sum(),(a[3]-a[3])*df1['price'].sum()]
          X=['30 days','60 days','90 days','365 days']
        elif X=="Name":
           X='name'
           Y='price'
        elif X=='property types':
          X='property_type'
          Y='price'
        elif X=='country':
          X='country'
          Y='price'
        fig1=px.pie(
          df1,
          values=Y,
          names=X,
          title="pie chart"
           )
       
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 

    if chart_type=='bar chart':
        if X=="seasons":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[(a[3]-a[0])*df1['price'].sum(),(a[3]-a[1])*df1['price'].sum(),(a[3]-a[2])*df1['price'].sum(),(a[3]-a[3])*df1['price'].sum()]
          X=['30 days','60 days','90 days','365 days']
        elif X=="Name":
           X='name'
           Y='price'
        elif X=='property types':
          X='property_type'
          Y='price'
        elif X=='country':
          X='country'
          Y='price'
        fig1=px.bar(
          df1,
          x=X,
          y=Y,
          title="Bar chart"
           )
       
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 
    if chart_type=='line chart':
        if X=="season":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[(a[3]-a[0])*df1['price'].sum(),(a[3]-a[1])*df1['price'].sum(),(a[3]-a[2])*df1['price'].sum(),(a[3]-a[3])*df1['price'].sum()]
          X=['365 days','90 days','60 days','30 days']
        elif X=="Name":
           X='name'
           Y='price'
        elif X=='property types':
          X='property_type'
          Y='price'
        elif X=='country':
          X='country'
          Y='price'
        fig1=px.line(
          df1,
          x=X,
          y=Y,
          title="line chart"
           )
       
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 

#This function display different types of chart for Visualization of availability
def avail_chart(df1,chart_type,X):
    if chart_type=='scatter chart':
        if X=="demand fluctuation in %":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[(((a[1]-a[0])*100)/a[0]),(((a[2]-a[1])*100)/a[1]),(((a[3]-a[2])*100)/a[2])]
          X=['30 days','60 days','90 days']
        elif X=="occupancy rates in %":
           X=['30 days','60 days','90 days']
           a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
           Y=[(a[3]-a[0])/a[3],(a[3]-a[0])/a[3],(a[3]-a[0])/a[3]]
        elif X=="avaliablity 30":
           X='availability_30'
           Y='price'
        elif X=="avaliablity 60":
           X='availability_60'
           Y='price'
        elif X=="avaliablity 90":
           X='availability_90'
           Y='price'
        elif X=="avaliablity 365":
           X='availability_365'
           Y='price'

        fig1=px.scatter(
          df1,
          x=X,
          y=Y,
          title="Scatter plots chart"
           )  
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 

    if chart_type=='line chart':
        if X=="demand fluctuation in %":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[(((a[1]-a[0])*100)/a[0]),(((a[2]-a[1])*100)/a[1]),(((a[3]-a[2])*100)/a[2])]
          X=['30 days','60 days','90 days']
        elif X=="occupancy rates in %":
           X=['30 days','60 days','90 days']
           a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
           Y=[(a[3]-a[0])/a[3],(a[3]-a[0])/a[3],(a[3]-a[0])/a[3]]
        elif X=="avaliablity 30":
           X='availability_30'
           Y='price'
        elif X=="avaliablity 60":
           X='availability_60'
           Y='price'
        elif X=="avaliablity 90":
           X='availability_90'
           Y='price'
        elif X=="avaliablity 365":
           X='availability_365'
           Y='price'
        fig1=px.line(
          df1,
          x=X,
          y=Y,
          title="line chart"
           )  
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 
    if chart_type=='bar chart':
        if X=="demand fluctuation in %":
          a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
          Y=[(((a[1]-a[0])*100)/a[0]),(((a[2]-a[1])*100)/a[1]),(((a[3]-a[2])*100)/a[2])]
          X=['30 days','60 days','90 days']
        elif X=="occupancy rates in %":
           X=['30 days','60 days','90 days']
           a=[df1['availability_30'].sum(),df1['availability_60'].sum(),df1['availability_90'].sum(),df1['availability_365'].sum()]
           Y=[(a[3]-a[0])/a[3],(a[3]-a[0])/a[3],(a[3]-a[0])/a[3]]
        elif X=="avaliablity 30":
           X='availability_30'
           Y='price'

        elif X=="avaliablity 60":
           X='availability_60'
           Y='price'
        elif X=="avaliablity 90":
           X='availability_90'
           Y='price'
        elif X=="avaliablity 365":
           X='availability_365'
           Y='price'
        fig1=px.bar(
          df1,
          x=X,
          y=Y,
          title="barchart"
           )  
        fig1.update_layout(hoverlabel_font_size=20,hoverlabel_font_family='Arial')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True,height=1780) 


st.title(":orange[Airbnb Analysis]")        
df=convert_mgcol_pd()
df=droping(df)
df=del_add_values(df)
option_dp1=list(df['country'].unique())
option_dp1.insert(0,'select all country')
co1, co2=st.columns(2)
ch1=co1.checkbox(label="map1")

dp1=co2.multiselect(label="select country",options=option_dp1)
co1a, co2a=st.columns(2)
dp2a=co1a.radio(label="select",options=["less than","more than"],horizontal=True)
dp2b=co2a.radio(label="select",options=["Daily rent","weekly rent","monthly rent"],horizontal=True)

co3, co4,co10=st.columns(3)
op2=list(df['property_type'].unique())
op2.insert(0,'select all')
if dp2b=="Daily rent":
  dp2=co3.selectbox(label="select Daily rent ",options=["select all",500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
elif dp2b=="weekly rent":
    dp2=co3.selectbox(label="select weekly rent ",options=["select all",5000,10000,15000,20000,25000,30000,35000])
else:
  dp2=co3.selectbox(label="select monthly rent ",options=["select all",5000,10000,15000,20000,25000,30000,35000,
                                                    40000,50000,60000,70000,80000,90000,100000,20000,30000,35000       ])
dp3=co4.selectbox(label="select bedrooms",options=["select all",0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
dp4=co10.selectbox(label="property type",options=op2)

try:
   map1,df1=map(df,dp1,dp2a,dp2,dp3,op2)
except :
   st.write("") 


if ch1==True and len(dp1)>0:
  try:
   st_data =folium_static(map1)
  except :
    st.write("") 
st.subheader(":violet[Price Visualization]")
co5,co6=st.columns(2)
chart_type=co5.selectbox(label="select a chart type",options=["select a chart","pie chart","line chart",'scatter chart','bar chart'])
X=co6.selectbox(label="select a value" ,options=['Name','property types',"country","season"])
st.subheader(":blue[avaliablity Visualization]")
co7,co8=st.columns(2)
chart_type1=co7.selectbox(label="select a charttype",options=["select a chart",'scatter chart','bar chart',"line chart"])
X1=co8.selectbox(label="select a value" ,options=['demand fluctuation in %','occupancy rates in %',"avaliablity 30",
                                                  "avaliablity 60","avaliablity 90","avaliablity 365"]) 
try:
  price_charts(df1,chart_type,X)
  avail_chart(df1,chart_type1,X1)
except:
  st.write("")   
