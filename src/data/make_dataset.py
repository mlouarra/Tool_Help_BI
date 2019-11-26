# -*- coding: utf-8 -*-
# import click
# import logging
from pathlib import Path
import json
import hunspell
import pandas as pd
from pandas.core.common import flatten
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

        df_malk['corrected_code'] = df_malk[['Material grp desc. 2', 'Matl Group']].apply(correction, axis=1)
        df_malk.dropna(inplace=True)
        return df_malk[~df_malk['Material grp desc. 2'].str.contains(pat="utiliser|Utiliser", regex=True)]

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

            if row.notnull()['corrected_code']:
                return row['corrected_code']
            else:
                return row['MATKL Material Group']
        df_merge['corrected_code'] = df_merge.apply(correction_test, axis=1)
        df_merge.drop_duplicates(inplace=True)
        df_merge = pd.merge(df_merge[['corrected_code', 'TXZ01 Short Text']], df_malk_correction[['corrected_code', 'Material grp desc. 2']], on='corrected_code', how='left')
        df_merge.drop_duplicates(inplace=True)
        df_merge.dropna(inplace=True)
        df_merge = df_merge[~df_merge['Material grp desc. 2'].str.contains('CTA A CHANGER')]
        df_merge['TXZ01 Short Text'] = df_merge['TXZ01 Short Text'].apply(lambda x: x.split(' '))
        df_merge = df_merge.groupby(['corrected_code', 'Material grp desc. 2'])['TXZ01 Short Text'].apply(
            list).reset_index(name='text_grouped')
        df_merge['text_grouped_flatten'] = df_merge['text_grouped'].apply(
            lambda x: list(flatten(x)))
        return df_merge

    def data_cleaned(self):
        pattern_digit = '[0-9]'
        def lemma(x):
            doc = nlp(x)
            list_test = []
            for token in doc:
                list_test.append(token.lemma_)
            return list_test

        df_to_clean = self.return_merged_df()
        stop_word = self._config['StopWord']
        df_to_clean['text_grouped_flatten'] = df_to_clean['text_grouped_flatten'].apply(
            lambda x: [element.lower() for element in x])
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(
            lambda x: [element.replace('"', '') for element in x])
        df_to_clean['text_grouped_flatten'] = df_to_clean['text_grouped_flatten'].apply(
            lambda x: [element.lower() for element in x])
        df_to_clean['text_grouped_flatten'] = df_to_clean['text_grouped_flatten'].apply(
            lambda x: [strip_multiple_whitespaces(element) for element in x])
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(
            lambda x: [strip_punctuation(element) for element in x])
        df_to_clean['text_grouped_flatten'] = df_to_clean['text_grouped_flatten'].apply(
            lambda x: [strip_tags(element) for element in x])
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(
            lambda x: [strip_non_alphanum(element) for element in x])
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(
            lambda x: [element for element in x if element not in stop_word])
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(
            lambda x: [re.sub(pattern_digit, "", element) for element in x])
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(lambda x: list(filter(None, x)))
        df_to_clean["text_grouped_flatten"] = df_to_clean["text_grouped_flatten"].apply(
            lambda x: [element for element in x if len(element) > 2])
        df_to_clean["text"] = df_to_clean["text_grouped_flatten"].apply(lambda x: ' '.join([element for element in x]))
        df_to_clean["text"] = df_to_clean['text'].apply(lemma)
        df_to_clean.rename(columns={'Material grp desc. 2': 'category'}, inplace=True)
        return df_to_clean

class DataToAzure:
    def __init__(self, df_cleaned, config):
        self._df_cleaned = df_cleaned
        self._config = config
    def json_to_azure(self):
        path_to_json = self._config['connect']['PathJson']
        df_json = self._df_cleaned[['corrected_code', 'category', 'text']].to_dict('records')
        # We save file
        with open(path_to_json, 'w') as fout:
            json.dump(df_json, fout)
