
                if(load_kedb(filename_raw, len(filename_raw)) == "sukses"):
                    filename_raw['label'] = filename_raw['label'].replace({'negatif': 0, 'positif':1})
                    # preprocess mulai
                    filename_raw["tweet_tokens_stemmed"] = text_preprocesing(filename_raw, "data-520")
                    # filename_raw["tweet_list"] = filename_raw["tweet_tokens_stemmed"].apply(convert_text_list)
                    # pre["tweet_list"] = pre["tweet_tokens_stemmed"].astype(str).apply(convert_text_list)

                    filename_raw["tweet_list"] = filename_raw["tweet_tokens_stemmed"].astype(str).apply(convert_text_list)
                    # bagi dataset train & testing
                    train = filename_raw.sample(frac=0.8, random_state=100)
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
                    # print(df_kamusdata['idf'].head())
                    # print(df_kamusdata.head())
                    # df_kamusdata['idf'].sum()

                    gk = train.groupby('label')
                    n_dok = gk.size()
                    # print(n_dok)

                    df_kamusdata_pos = gk.get_group(1)
                    DF_pos = calc_DF(df_kamusdata_pos["TF_dict"])
                    IDF_pos = calc_IDF(n_document, DF_pos)
                    df_kamusdata_posnya = pd.DataFrame(list(DF_pos.items()),columns = ['term','tf'])
                    df_kamusdata_posnya["idf"] = pd.DataFrame(list(IDF_pos.values()),columns = ['idf'])
                    df_kamusdata_posnya["TF-IDF_dict"] = df_kamusdata_posnya.tf * df_kamusdata_posnya.idf
                    # df_kamusdata_posnya["TF-IDF_dict"].sum()

                    df_kamusdata_neg = gk.get_group(0)
                    DF_neg = calc_DF(df_kamusdata_neg["TF_dict"])
                    IDF_neg = calc_IDF(n_document, DF_neg)
                    df_kamusdata_negnya = pd.DataFrame(list(DF_neg.items()),columns = ['term','tf'])
                    df_kamusdata_negnya["idf"] = pd.DataFrame(list(IDF_neg.values()),columns = ['idf'])
                    df_kamusdata_negnya["TF-IDF_dict"] = df_kamusdata_negnya.tf * df_kamusdata_negnya.idf
                    # df_kamusdata_negnya["TF-IDF_dict"].sum()

                    # prepare table insert (delete)
                    pre_loaddata('dataset_procesed')
                    pre_loaddata('dataset_training')
                    pre_loaddata('dataset_tes')
                    pre_loaddata('kamus_data_all')
                    pre_loaddata('kamus_positif')
                    pre_loaddata('kamus_negatif')

                    conn = mysql.connect()
                    cursor = conn.cursor()

                    # Insert data_preproces
                    query = """INSERT INTO
                                dataset_procesed (
                                    label,
                                    tweet_tokens_stemmed)
                            VALUES(%s, "%s");"""

                    i=0
                    while i < len(filename_raw):
                        cursor.execute(query, (filename_raw['label'].iloc[i], filename_raw['tweet_tokens_stemmed'].iloc[i]))
                        conn.commit()
                        i+=1

                    # Insert data Training & Testing
                    # Insert data_preproces
                    query = """INSERT INTO
                                dataset_tes (
                                    label,
                                    tweet_tokens_stemmed)
                            VALUES(%s, "%s");"""

                    i=0
                    while i < len(test):
                        cursor.execute(query, (test['label'].iloc[i], test['tweet_list'].iloc[i]))
                        conn.commit()
                        i+=1


                    # Insert data Training & Testing
                    # Insert data_preproces
                    query = """INSERT INTO
                                dataset_training (
                                    label,
                                    tweet_tokens_stemmed)
                            VALUES(%s, "%s");"""

                    i=0
                    while i < len(train):
                        cursor.execute(query, (train['label'].iloc[i], train['tweet_list'].iloc[i]))
                        conn.commit()
                        i+=1



                    # Insert kamus_data_all
                    query = """INSERT INTO
                                kamus_data_all (
                                    term,
                                    df,
                                    idf)
                            VALUES(%s, %s, %s);"""

                    i=0
                    while i < len(df_kamusdata):
                        cursor.execute(query, ( df_kamusdata['term'][i], df_kamusdata['df'][i], df_kamusdata['idf'][i], ))
                        conn.commit()
                        i+=1


                    # Insert kamus_positif
                    query = """INSERT INTO
                                kamus_positif (
                                    term,
                                    tf,
                                    idf,
                                    tf_idf_dict)
                            VALUES(%s, %s ,%s ,%s);"""

                    conn = mysql.connect()
                    cursor = conn.cursor()

                    i=0
                    while i < len(df_kamusdata_posnya):
                        cursor.execute(query, (df_kamusdata_posnya['term'][i], df_kamusdata_posnya['tf'][i], df_kamusdata_posnya['idf'][i], df_kamusdata_posnya['TF-IDF_dict'][i], ))
                        conn.commit()
                        i+=1


                    # Insert kamus_negatif
                    query = """INSERT INTO
                                kamus_negatif (
                                    term,
                                    tf,
                                    idf,
                                    tf_idf_dict)
                            VALUES(%s ,%s ,%s,%s);"""

                    conn = mysql.connect()
                    cursor = conn.cursor()

                    i=0
                    while i < len(df_kamusdata_negnya):
                        cursor.execute(query, (df_kamusdata_negnya['term'][i], df_kamusdata_negnya['tf'][i], df_kamusdata_negnya['idf'][i], df_kamusdata_negnya['TF-IDF_dict'][i], ))
                        conn.commit()
                        i+=1

                    conn.close()

                    msg="Data berhasil ditambahkan.."

                return redirect(url_for('dataset', username=session['username'], msg=msg))