# -*- coding: utf-8 -*-
# import click
# import logging
from pathlib import Path
import json
import hunspell
import pandas as pd
from unidecode import unidecode
from gensim.parsing.preprocessing import split_alphanum
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import spacy
import re
nlp = spacy.load("fr_core_news_sm")

class DataProcessing:

    def __init__(self, config):
        self._config = config

    def list_ekpo_files(self):
        return sorted(Path(self._config['connect']['PathText']).glob("*EKPO*"))

    def df_malk_correction(self):
        regexp = re.compile('utiliser ', re.IGNORECASE)
        df_malk = pd.read_pickle(self._config['connect']['path_file_fr_cleaned'])
        df_malk['Matl Group'] = df_malk['Matl Group'].astype(str)
        df_malk['Material grp desc. 2'] = df_malk['Material grp desc. 2'].astype(str)

        def correction(row):
            if regexp.search(str(row['Material grp desc. 2'])):
                temp = regexp.sub('', row['Material grp desc. 2'])
                temp.strip()
                return temp
            else:
                return row['Matl Group']

        df_malk['code_correction'] = df_malk[['Material grp desc. 2', 'Matl Group']].apply(correction, axis=1)
        df_malk.dropna(inplace=True)
        return df_malk[~df_malk['Material grp desc. 2'].str.contains(pat ="utiliser|Utiliser", regex=True)]

    def return_merged_df(self, ):
        list_ekpo = self.list_ekpo_files()
        df_list = []
        for i in range(len(list_ekpo)):
            df_list.append((pd.read_csv(str(list_ekpo[i]), sep=';', error_bad_lines=False, dtype='unicode')))

        df_total = pd.concat(df_list)
        del df_list

        df_correction = pd.read_pickle(self._config['connect']['PathCodeCorrige'])
        df_correction = df_correction.astype(str)
        df_correction.rename(columns={'Matl Group': 'MATKL Material Group'}, inplace=True)
        df_malk_correction = self.df_malk_correction()
        df_malk_correction = df_malk_correction.astype(str)
        df_merge = pd.merge(df_total, df_correction, on='MATKL Material Group', how='left')


        def correction_test(row):

            if row.notnull()['code_correction']:
                return row['code_correction']
            else:
                return row['MATKL Material Group']

        df_merge['code_correction'] = df_merge.apply(correction_test, axis=1)
        df_merge.drop_duplicates(inplace=True)
        df_util = pd.merge(df_merge[['code_correction', 'TXZ01 Short Text']], df_malk_correction[['code_correction', 'Material grp desc. 2']], on='code_correction', how='left')
        del df_merge
        df_util.drop_duplicates(inplace=True)
        df_util.dropna(inplace=True)
        df_util = df_util[~df_util['Material grp desc. 2'].str.contains('CTA A CHANGER')]

        return df_util

    def data_cleaned(self):
        def lemma(x):
            doc = nlp(x)
            list_test = []
            for token in doc:
                list_test.append(token.lemma_)
            return list_test
        stop_word = self._config['StopWord']
        df_clean = self.return_merged_df()
        df_clean.drop_duplicates(inplace=True)
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].astype(str)
        df_clean['Material grp desc. 2'] = df_clean['Material grp desc. 2'].astype(str)
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].str.lower()
        df_clean["TXZ01 Short Text"] = df_clean["TXZ01 Short Text"].apply(lambda x: x.replace('"', ''))
        df_clean["Material grp desc."] = df_clean["Material grp desc. 2"].apply(lambda x: x.replace('"', ''))
        df_clean['Material grp desc. 2'] = df_clean['Material grp desc. 2'].apply(lambda x: strip_multiple_whitespaces(x))
        #df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: unidecode(x))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: split_alphanum(x))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: strip_punctuation(x))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: strip_tags(x))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: strip_non_alphanum(x))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: strip_multiple_whitespaces(x))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_word]))
        df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].str.replace("\d+", '')
        # df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lemma)
        # df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: [element for element in x if element.strip()])
        # df_clean['TXZ01 Short Text'] = df_clean['TXZ01 Short Text'].apply(lambda x: [element for element in x if len(element)>2])
        df_clean.rename(columns={"Material grp desc. 2": "category"}, inplace=True)
        df_clean = df_clean[~df_clean['category'].str.contains('CTA A CHANGER')]
        return df_clean

class DataToAzure:
    def __init__(self, df_cleaned, config):
        self._df_cleaned = df_cleaned
        self._config = config

    def json_to_azure(self, grouped=True):
        path_to_json = self._config['connect']['PathJson']
        if grouped:
            df_json = self._df_cleaned.groupby(['code_correction', 'category'])['TXZ01 Short Text'].apply(
                list).reset_index(name='text_grouped')
            df_json['text_grouped'] = df_json['text_grouped'].apply(
                lambda x: [item for sublist in x for item in sublist])
            df_json = df_json[["text_grouped", "code_correction", "category"]].to_dict('records')
        # We save file
            with open(path_to_json, 'w') as fout:
                json.dump(df_json, fout)
        else:
            pass
            # with open(path_to_json, 'w') as fout:
            #     json.dump(self._df_cleaned[["TXZ01 Short Text", "code_corrige", "Long"]].to_dict('records'), fout)

