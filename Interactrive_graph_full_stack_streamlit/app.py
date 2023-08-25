import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime
import snowflake.connector as sf
import numpy as np
from GoogleNews import GoogleNews
from datetime import timedelta
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()
import json
import plotly.express as px
import plotly.graph_objects as go

def views_prediction2(string):
    # create a connection object
    conn = sf.connect(
        user='ankan',
        password='ankanProboReset@123',
        account='rl48423.ap-south-1.aws',
        database='MELTANO_DB',
        schema='PROBO_ANALYTICS'
    )
    cursor = conn.cursor()
    cursor.execute(string)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
    return df

string = '''
with right_events as (select distinct event_id ids from (select event_id,max(created_dt) max_dt,min(created_dt) min_dt,
datediff(days,min_dt,max_dt) dt_diff from 
(select event_id,created_dt, datediff(days ,created_dt, current_timestamp()) days_old 
from probo_analytics.tms_trade having days_old<=180)tb1 group by event_id having dt_diff>=30)tb2)

select table_2.NAME,table_2.description,table_1.event_id,table_1.CREATED_DT,table_1.BUY_PRICE from 
(select event_id,CREATED_DT,BUY_PRICE from PROBO_ANALYTICS.tms_trade where event_id  in (
select distinct event_id from (
select event_id, min(avg_daily) min_price,max(avg_daily) max_price,max_price-min_price price_diff from
(select event_id,dt,avg(buy_price) avg_daily from 
(select event_id,cast(created_dt as date) dt,buy_price from 
probo_analytics.tms_trade where event_id in (select ids from right_events))t group by event_id,dt)t2 
group by event_id having price_diff>=6.5)tb3
)) as table_1
left join probo_analytics.events as table_2 on table_1.event_id = table_2.id ;'''

df_ori = views_prediction2(string)

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    # lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

def get_news(time_stamp,title,gap = 60):
    stop_words = set(stopwords.words('english'))
    processed=[]
    line = title
    words = line.split()
    for r in words:
        if not r in stop_words:
            processed.append(r)
    string = " ".join(processed)
    string = preprocess(string)
    time_stamp = datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
    yesterday = time_stamp - timedelta(minutes=gap)
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    googlenews=GoogleNews(start=yesterday.strftime("%m-%d-%Y"),end=time_stamp.strftime("%m-%d-%Y"))
    googlenews.get_news(string)
    result=googlenews.result()
    df=pd.DataFrame(result)
    return df

def process(input_value,df_ori):
    df = df_ori.drop(['Unnamed: 0'], axis='columns')
    
    event_759242 = df.loc[df['EVENT_ID'] == input_value].sort_values('CREATED_DT')
    original_table = eval('event_759242')
    Table = original_table
    Table['rate_change'] = Table['BUY_PRICE'].pct_change(fill_method='ffill')
    Table['get_data_change'] = np.where(abs(Table['rate_change'])>=Table['rate_change'].std() ,Table['BUY_PRICE'],None)    
    Table['get_data'] = np.where(Table['get_data_change'] ,True,False)
    Table.reset_index(drop=True,inplace=True)
    
    print( str('10 percent'))
    
    time_stamp = []
    time_stamp = Table.query('get_data == True')['CREATED_DT']
    title = Table.query('get_data == True')['NAME']
    description = Table.query('get_data == True')['DESCRIPTION']

    time_stamp = time_stamp.tolist()
    title = title.tolist()
    description = description.tolist()
    Table.loc[Table['get_data'] == True, 'Tickker'] = Table['BUY_PRICE']
    Table['News'] = np.nan
    Table['news_link'] = np.nan
    Table['Sentiment'] = np.nan
    news_list = []
    link_list = []
    sent_list = []
    
    print( str('20 percent'))
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    size = len(Table['BUY_PRICE'])
    for i in range(int((size))):
        my_bar.progress(i/size, text=progress_text)
        if Table['get_data'][i] == True:
            news = get_news(Table['CREATED_DT'][i] ,Table['NAME'][i])
            try:
                for k in range(len(news['title'])):
                    if news['title'][k] not in news_list :
                        news_list.append(news['title'][k])
                        link_list.append(news['link'][k])
                        Table['News'][i] = news['title'][k]
                        Table['news_link'][i] = news['link'][k]
                        sent_1 = sentiment.polarity_scores(news['title'][k])
                        sent_list.append(sent_1['compound'])
                        Table['Sentiment'][i] = sent_1['compound']
                    else:
                        Table['News'][i] = news_list[-1]
                        Table['news_link'][i] = link_list[-1]
                        Table['Sentiment'][i] = sent_list[-1]
            except:
                pass
    my_bar.progress(100, text=progress_text)

            

    # new_Table = Table.drop(['EVENT_ID',"rate_change",'BUY_PRICE','get_data_change','get_data','Tickker','DESCRIPTION'], axis='columns')
    # new_Table = new_Table.dropna()
    # final_table = pd.merge(original_table, new_Table, on='CREATED_DT')
    # final_table = Table.drop(['NAME',"DESCRIPTION",'EVENT_ID','get_data_change','get_data_change','get_data','Tickker',
    #                         'News','news_link','Sentiment'], axis='columns')
    # final_table = pd.merge(final_table,new_Table,how ='outer',on='CREATED_DT')
    # final_table = final_table.drop(['rate_change',"NAME"], axis='columns')
    
    # return json.dumps(final_table.to_dict(), indent = 4) 
    return Table

with st.form("my_form"):
    option = st.selectbox(
        'Please select the event',
        np.sort(df_ori.NAME.unique()))
    id_name={}
    for i in range(len(df_ori.EVENT_ID.unique())):
        id_name[df_ori.NAME.unique()[i]] = df_ori.EVENT_ID.unique()[i]
        
    submitted = st.form_submit_button("Submit")
    if submitted:
        Table = process(id_name[option],df_ori)
        st.subheader("Define a custom colorscale")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Table['CREATED_DT'],
                                y=Table['BUY_PRICE'],
                                mode='lines',
                                name='price'
                                )
                    )

        fig.add_trace(go.Scatter(x=Table['CREATED_DT'],
                                y=Table['Tickker'],
                                mode='markers+text',
                                text=Table['News'],
                                name=''))
        fig.update_traces(marker_size=10,textfont_size=1)
        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with tab2:
            st.plotly_chart(fig, theme=None, use_container_width=True)
    else:
        pass

    
