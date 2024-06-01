from flask import Flask, render_template, request, Response, make_response, jsonify, redirect, url_for, send_file
from whitenoise import WhiteNoise
import pandas as pd
import os.path as path
from dkrhtextpreprocesing import text_preprocesing
from tf_idf import *
import csv, json, time
from count import CountDF
import numpy as np
from nltk.tokenize import word_tokenize
from model import Vectorize, Model
from sklearn.model_selection import train_test_split
from genplot import *
from naive_bayes import *

df = 'dataset/ori/df.csv'
PATH= {
    'NORMALFILE' : 'dataset/setting/normalisasi.csv',
    'STOPWORDS' : 'dataset/setting/stopwords.csv',
    'EMOTICON' : 'dataset/setting/emoticon.csv',
}
#-- Web Based Start --#
app = Flask(__name__)

app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")

@app.route('/', methods=['POST', 'GET'])
def main():
    return redirect('/dataset', 302)

@app.route('/dataset', methods=['POST', 'GET'])
def dataset():
    _dr = '';
    _dp = '';
    if path.exists('data/pre/all.csv'):
        _dr = pd.read_csv('data/pre/all.csv',sep='`')
        _dr = _dr.to_html(classes="table")

    if path.exists('data/post/dataset_procesed.csv'): 
        _dp = pd.read_csv('data/post/dataset_procesed.csv',sep='`', usecols=['tweet_tokens_stemmed','label_name'])
        _dp = _dp.to_html(classes="table")

    return render_template('dataset.html', dr=_dr, dp=_dp, msg="sukses")

@app.route('/dataset/upload', methods=['POST', 'GET'])
def datasetupload():
    load_dataset_processed()
    #load_count_word()
    chi()
    chi_after()
    load_json()

def load_dataset_processed():
    ef = pd.read_csv('data/pre/all.csv',sep='`')
    filename_raw = pd.DataFrame(ef, columns=['tweets','label'])
    filename_raw['label_name'] = filename_raw['label'].str.lower()
    filename_raw['label'] = filename_raw['label'].str.lower().replace({'negatif': 0, 'positif':1, 'netral':2})
    
    v = text_preprocesing(filename_raw)
    filename_raw["tweet_tokens_stemmed"] = v[0]
    filename_raw["text_token"] = v[1]
    filename_raw["tweet_list"] = filename_raw["tweet_tokens_stemmed"].astype(str).apply(convert_text_list)

    train = filename_raw.sample(frac=0.7, random_state=100)
    # train = filename_raw.sample(frac=0.8)
    test = filename_raw[~filename_raw.index.isin(train.index)]
    
    # Pembobotan tf-idf train
    train["TF_dict"] = train['tweet_list'].apply(calc_TF)
    DF = calc_DF(train["TF_dict"])
    # print(len(DF))
    n_document = len(train)
    IDF = calc_IDF(n_document, DF)

    df_kamusdata = pd.DataFrame(list(IDF.items()),columns = ['term','idf'])
    df_kamusdata['df'] = pd.DataFrame(list(DF.values()), columns = ['df'])
    gk = train.groupby('label')
    #print(gk.first())
    n_dok = gk.size()
    # print(n_dok)
    df_kamusdata_pos = gk.get_group(1)
    DF_pos = calc_DF(df_kamusdata_pos["TF_dict"])
    IDF_pos = calc_IDF(n_document, DF_pos)
    df_kamusdata_posnya = pd.DataFrame(list(DF_pos.items()),columns = ['term','tf'])
    df_kamusdata_posnya["idf"] = pd.DataFrame(list(IDF_pos.values()),columns = ['idf'])
    df_kamusdata_posnya["TF-IDF_dict"] = df_kamusdata_posnya.tf * df_kamusdata_posnya.idf

    df_kamusdata_neg = gk.get_group(0)
    DF_neg = calc_DF(df_kamusdata_neg["TF_dict"])
    IDF_neg = calc_IDF(n_document, DF_neg)
    df_kamusdata_negnya = pd.DataFrame(list(DF_neg.items()),columns = ['term','tf'])
    df_kamusdata_negnya["idf"] = pd.DataFrame(list(IDF_neg.values()),columns = ['idf'])
    df_kamusdata_negnya["TF-IDF_dict"] = df_kamusdata_negnya.tf * df_kamusdata_negnya.idf

    df_kamusdata_net = gk.get_group(2)
    DF_net = calc_DF(df_kamusdata_net["TF_dict"])
    IDF_net = calc_IDF(n_document, DF_net)
    df_kamusdata_netnya = pd.DataFrame(list(DF_net.items()),columns = ['term','tf'])
    df_kamusdata_netnya["idf"] = pd.DataFrame(list(IDF_net.values()),columns = ['idf'])
    df_kamusdata_netnya["TF-IDF_dict"] = df_kamusdata_netnya.tf * df_kamusdata_netnya.idf

    vz = np.concatenate(filename_raw['tweet_tokens_stemmed'])
    vv = filename_raw.groupby(['label_name']).agg({'text_token': ' '.join})
    vv = pd.DataFrame(vv)
    
    filename_result1 = pd.DataFrame(filename_raw, columns=['label','label_name','tweet_tokens_stemmed','text_token'])
    filename_result2 = pd.value_counts(np.array(vz)).rename_axis('label').reset_index(name='counts')
    filename_result3 = vv
    
    datatoexcel1 = 'data/post/dataset_procesed.csv'
    datatoexcel2 = 'data/post/dataset_counts.csv'
    datatoexcel3 = 'data/post/dataset_counts2.csv'
    filename_result1.to_csv(datatoexcel1, sep='`')
    filename_result2.to_csv(datatoexcel2, sep='`')
    filename_result3.to_csv(datatoexcel3, sep='`')

    train.to_csv('data/post/dkrh_train.csv', sep='`')
    test.to_csv('data/post/dkrh_test.csv', sep='`')
    df_kamusdata.to_csv('data/post/dkrh_dict.csv', sep='`')
    df_kamusdata_posnya.to_csv('data/post/dkrh_dictpos.csv', sep='`')
    df_kamusdata_negnya.to_csv('data/post/dkrh_dictneg.csv', sep='`')
    df_kamusdata_netnya.to_csv('data/post/dkrh_dictnet.csv', sep='`')

def load_count_word():
    ef = pd.read_csv('data/post/dataset_counts2.csv',sep='`')
    filename_post = pd.DataFrame(ef)
    dp = filename_post['text_token'].loc[filename_post['label_name'] == 'positif'].split()
    r1 = pd.value_counts(np.array(dp)).rename_axis('label').reset_index(name='counts')

    datatoexcel3 = 'data/post/dataset_counts3.csv'
    r1.to_csv(datatoexcel3, sep='`')

def chi():
    df = pd.read_csv('data/post/dataset_procesed.csv',sep='`')
    out = df['text_token'].str.split().explode().to_frame('text').join(df['label_name']).assign(value=1)
    out = out.pivot_table('value', 'text', 'label_name', aggfunc='count', fill_value=0)
    out = out.assign(count=lambda x: x.sum(axis=1))
    out = out.assign(evpos=0.0)
    out = out.assign(evneg=0.0)
    out = out.assign(evnet=0.0)
    out = out.assign(evtot=0.0)
    out = out.assign(chipos=0.0)
    out = out.assign(chineg=0.0)
    out = out.assign(chinet=0.0)
    out = out.assign(chitot=0.0)
    totpos = float(out['positif'].sum())
    totneg = float(out['negatif'].sum())
    totnet = float(out['netral'].sum())
    totall = float(out['count'].sum())
    #print(totpos,totneg,totnet,totall, sep='-')
    i = len(df.index)
    for i, row in out.iterrows():
        countz = float(out.at[i,'count'])
        opos = float(out.at[i,'positif'])
        oneg = float(out.at[i,'negatif'])
        onet = float(out.at[i,'netral'])
        evpos = float((countz*totpos)/totall)
        evneg = float((countz*totneg)/totall)
        evnet = float((countz*totnet)/totall)
        evtot = evpos+evneg+evnet
        #print(countz,totpos,totall, sep='**')
        #print(evpos,evneg,evnet,evtot, sep='_')
        out.at[i,'evpos'] = evpos
        out.at[i,'evneg'] = evneg
        out.at[i,'evnet'] = evnet
        out.at[i,'evtot'] = evtot
        out.at[i,'chipos'] = chifs(opos,evpos)
        out.at[i,'chineg'] = chifs(oneg,evpos)
        out.at[i,'chinet'] = chifs(onet,evpos)
        out.at[i,'chitot'] = out.at[i,'chipos']+ out.at[i,'chineg']+out.at[i,'chinet']
        print(out.at[i,'chitot'])
    out.to_csv('data/post/dataset_chi.csv', sep='`')

def chi_after():
    df = pd.read_csv('data/post/dataset_chi.csv',sep='`')
    res = []
    for i, row in df.iterrows():
        if df.at[i,'chitot'] < 3.841:
            df.drop(i)
        else:
            if (df.at[i,'positif'] > 0):
                text = [ df.at[i,'text'] ] * int(df.at[i,'positif'])
                res.append(['positif',text,df.at[i,'text'],int(df.at[i,'positif']),])
            elif (df.at[i,'negatif'] > 0):
                text = [ df.at[i,'text'] ] * int(df.at[i,'negatif'])
                res.append(['negatif',text,df.at[i,'text'],int(df.at[i,'negatif']),])
            elif (df.at[i,'netral'] > 0):
                text = [ df.at[i,'text'] ] * int(df.at[i,'netral'])
                res.append(['netral',text,df.at[i,'text'],int(df.at[i,'netral']),])
    #print(res)
    out = pd.DataFrame(res,columns=['label','text','a1','a2'])
    df.to_csv('data/post/dataset_chi_after.csv', sep='`')
    out.to_csv('data/post/dataset_chi_after2.csv', sep='`')

def load_json():
    start_time = time.time()
    result = pd.read_csv('data/post/dataset_chi_after2.csv',sep='`')
    tfidf_train = Vectorize(result['text'])
    tfidf_train.export()
    text_feature = tfidf_train.get_transform()
    text_feature_label = result['label']
    X_train, X_test, y_train, y_test = train_test_split(text_feature, text_feature_label, random_state=1, test_size=0.2)
    model = Model(X_train, X_test, y_train, y_test)
    model.export()
    time_result = round((time.time() - start_time)/60,2)
    return jsonify({
        'status' : 'Export Model and TFIDF Success',
        'training_time' : time_result,
    })

def chifs(o,e):
    #print(o,e,((o-e)**2.0)/e, sep='_')
    return float(((o-e)**2.0)/e)

def count_words_label(texts, words=10):
    texts = texts.dropna()
    texts = texts.reset_index(drop=True)
    texts = texts.apply(word_tokenize)
    bow, dict_words = {}, {}
    bow_sort = None
    for text in texts:
        for token in text:
            if token not in bow.keys():
                bow[token] = 1
            else:
                bow[token] += 1
    if words == 'all':
        bow_sort = sorted(bow.items(), key=lambda x: x[1], reverse=True)
    else:
        bow_sort = sorted(bow.items(), key=lambda x: x[1], reverse=True)[:words]
    for data in bow_sort:
        dict_words[data[0]] = data[1]
    return dict_words

@app.route('/chi-square-fs', methods=['GET'])
def chisquarefs():
    _dr = '';
    _dp = '';
    _dz = '';
    if path.exists('data/post/dataset_counts.csv'):
        _dr = pd.read_csv('data/post/dataset_counts.csv',sep='`', usecols=['label','counts'])
        _dr = _dr.to_html(classes="table")

    if path.exists('data/post/dataset_chi.csv'):
        _dp = pd.read_csv('data/post/dataset_chi.csv',sep='`')
        _dp = _dp.to_html(classes="table")

    if path.exists('data/post/dataset_chi_after.csv'):
        _dz = pd.read_csv('data/post/dataset_chi_after.csv',sep='`')
        _dz = _dz.to_html(classes="table")

    return render_template('chi.html', dr=_dr, dp=_dp, dz=_dz, msg="sukses")

@app.route('/naive-bayes2', methods=['GET'])
def naivebayes2():
    file_json = open('data/post/nb.json', 'r')
    data = json.load(file_json)
    file_json.close()
    _dr = '';
    wcpos = '';
    wcneg = '';
    wcnet = '';
    if path.exists('data/post/dataset_chi_after2.csv'):
        _dz = pd.read_csv('data/post/dataset_chi_after2.csv',sep='`')
        dz1 = _dz.loc[_dz['label'] == 'positif']
        dz2 = _dz.loc[_dz['label'] == 'negatif']
        dz3 = _dz.loc[_dz['label'] == 'netral']
        dz1.to_csv('data/post/dataset_1231.csv', sep='`')
        dz2.to_csv('data/post/dataset_1232.csv', sep='`')
        dz3.to_csv('data/post/dataset_1233.csv', sep='`')
        wcpos=get_wordcloud(dict(_dz.loc[_dz['label'] == 'positif'][['a1','a2']].values))
        wcneg=get_wordcloud(dict(_dz.loc[_dz['label'] == 'negatif'][['a1','a2']].values))
        wcnet=get_wordcloud(dict(dz3[['a1','a2']].values))
        data_json = data['model_reports']['cfm']
        
        dr = pd.DataFrame(data_json,columns=['Positif', 'Negatif', 'Netral'])
        dr.insert(0, "Label", ['Positif', 'Negatif', 'Netral'], True)
        dr = dr.to_html(classes="table")
    return render_template("nb.html", 
                                    a1=data, 
                                    dr=dr, 
                                    msg="sukses",
                                    img_data_pos=wcpos,
                                    img_data_neg=wcneg,
                                    img_data_net=wcnet
                            )

@app.route('/naive-bayes', methods=['GET'])
def naivebayes():
    a1 = pd.read_csv('data/post/dataset_procesed.csv',sep='`', usecols=['label_name'])
    a2 = pd.read_csv('data/post/dkrh_train.csv',sep='`')
    a3 = pd.read_csv('data/post/dkrh_test.csv',sep='`')
    a4 = pd.read_csv('data/post/dkrh_dict.csv',sep='`')
    a5 = pd.read_csv('data/post/dkrh_dictpos.csv',sep='`')
    a6 = pd.read_csv('data/post/dkrh_dictneg.csv',sep='`')
    a7 = pd.read_csv('data/post/dkrh_dictnet.csv',sep='`')

    datahasil = {}
    datahasil = naive_bayes(a3, a2, a5, a6, a7, a4)

    show = pd.DataFrame(datahasil)
    #show.to_csv('data/post/dkrhtes.csv', sep='`')
    show['Label_Prediksi'] = show['Label_Prediksi'].replace({'Negatif': 0, 'Positif':1, 'Netral':2})
    show["label_asal"]=[x for x in a3["label"]]

    df_matrik = pd.DataFrame(show, columns=['label_asal','Label_Prediksi'])
    # print(df_matrik)
    # df_matrik = pd.DataFrame(show, columns=['y_Actual','y_Predicted'])

    confusion_matrix = pd.crosstab(df_matrik['label_asal'], df_matrik['Label_Prediksi'], rownames=['Actual'], colnames=['Predicted'])
    # print(confusion_matrix)
    # buat visualisasi data
    diagram_dataset_pie=piechart(a1)
    kemunculan_kata=barchart(a4[["term","df"]])
    # kemunculan_kata=barchart(df_kamusdata[["term","df"]])
    # word_cloud_neg=get_wordcloud(df_kamusdata_negnya)
    word_cloud_pos=get_wordcloud(dict(a5[['term','TF-IDF_dict']].values))
    word_cloud_neg=get_wordcloud(dict(a6[['term','TF-IDF_dict']].values))
    word_cloud_net=get_wordcloud(dict(a7[['term','TF-IDF_dict']].values))
    #print(confusion_matrix[0])
    return render_template("nb.html",
                                    show=show.to_html(classes="table"),
                                    confusion_matrix=confusion_matrix.to_html(classes="table"),
                                    akurasi=round(akurasi(confusion_matrix), 2) * 100,
                                    pr=precisionrecall(confusion_matrix).to_html(classes="table"),
                                    barJSON=kemunculan_kata,
                                    pieJSON=diagram_dataset_pie,
                                    img_data_pos=word_cloud_pos,
                                    img_data_neg=word_cloud_neg,
                                    img_data_net=word_cloud_net,
                                    msg='ok')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template("dashboard.html", msg="sukses")

@app.route('/api/count/labels', methods=['POST'])
def api_counts_labels():
    if request.method == 'POST':
        count = CountDF('data/pre/all.csv')
        count_result = count.count_labels(labels='label')
        count_json = {
            'index' : count_result.index.values.tolist(),
            'values' : count_result.values.tolist(),
            'length_df' : count.length_df(),
        }
        return jsonify(count_json)
    else:
        return jsonify({'result' : 'error'})

@app.route('/api/count/words', methods=['POST'])
def api_counts_words():
    if request.method == 'POST':
        df = pd.read_csv('data/post/dataset_procesed.csv',sep='`')
        count = CountDF('data/post/dataset_procesed.csv')
        count_result = count.count_words(labels='text_token', words=10)
        count_result_positive = count.count_words_label(df['text_token'].loc[df['label_name'] == 'positif'], words=10)
        count_result_neutral = count.count_words_label(df['text_token'].loc[df['label_name'] == 'netral'], words=10)
        count_result_negative = count.count_words_label(df['text_token'].loc[df['label_name'] == 'negatif'], words=10)
        count_json = {
            'index' : list(count_result.keys()),
            'values' : list(count_result.values()),
            'length_df' : count.length_df(),
            'positive_words' : {
                'label' : list(count_result_positive.keys()),
                'value' : list(count_result_positive.values()),
            },
            'neutral_words' : {
                'label' : list(count_result_neutral.keys()),
                'value' : list(count_result_neutral.values()),
            },
            'negative_words' : {
                'label' : list(count_result_negative.keys()),
                'value' : list(count_result_negative.values()),
            }
        }
        return jsonify(count_json)
    else:
        return jsonify({'result' : 'error'})

@app.route('/api/info', methods=['GET'])
def api_get_info():
    if request.method == 'GET':
        arr_stop = []
        count = CountDF('data/pre/all.csv')
        #Get All Words
        get_all_words = count.count_words(labels='tweets', words='all')
        # Get All Stopwords
        txt_stopword = pd.read_csv('data/setting/stopwords.txt', names=['stopwords_id'], header=None)
        arr_stop.append(txt_stopword['stopwords_id'][0].split(' '))
        # Get Normalization Data
        load_word = pd.read_excel('data/setting/normalisasi.xlsx')
        return jsonify({
            'all_words' : len(get_all_words.keys()),
            'total_stopwords' : len(arr_stop[0]),
            'total_normalization' : len(load_word),
        })

if __name__ == '__main__':
    app.run(debug=True, port=33507)
#-- Web Based End --#
