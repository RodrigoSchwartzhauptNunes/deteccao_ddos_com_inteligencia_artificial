################################
#By: Rodrigo Schwartzhaupt Nunes
################################
#LOG
#10/09/2023 23:37
# Estou renomeando as colunas recebidas para terem os mesmo nome estarem na mesma ordem das de treinamento. 


import time
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


####### ARGs ####### 
funcao = "realtime"
####################
########## CAPTURA EM TEMPO REAL ##########
if funcao == "realtime":

    # Carregar os modelos treinados a partir dos arquivos
    modelo_floresta_aleatoria = joblib.load('modelo_floresta_aleatoria.pkl')
    modelo_arvore_decisao = joblib.load('modelo_arvore_decisao.pkl')

    def tomar_acoes(resultado):
        if resultado == 1:
            print("Ação tomada: Possível anomalia detectada")
        else:
            print("Tudo parece normal")

    def analisar_trafego(dados, modelo):
        previsoes = modelo.predict(dados)
        return previsoes


    mapeamento_colunas = {
    'src_port': ' Source Port',
    'dst_port': ' Destination Port',
    'protocol': ' Protocol',
    'flow_duration': ' Flow Duration',
    'tot_fwd_pkts': ' Total Fwd Packets',
    'tot_bwd_pkts': ' Total Backward Packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packets',
    'totlen_bwd_pkts': ' Total Length of Bwd Packets',
    'fwd_pkt_len_max': ' Fwd Packet Length Max',
    'fwd_pkt_len_min': ' Fwd Packet Length Min',
    'fwd_pkt_len_mean': ' Fwd Packet Length Mean',
    'fwd_pkt_len_std': ' Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': ' Bwd Packet Length Min',
    'bwd_pkt_len_mean': ' Bwd Packet Length Mean',
    'bwd_pkt_len_std': ' Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': ' Flow Packets/s',
    'flow_iat_mean': ' Flow IAT Mean',
    'flow_iat_std': ' Flow IAT Std',
    'flow_iat_max': ' Flow IAT Max',
    'flow_iat_min': ' Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': ' Fwd IAT Mean',
    'fwd_iat_std': ' Fwd IAT Std',
    'fwd_iat_max': ' Fwd IAT Max',
    'fwd_iat_min': ' Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': ' Bwd IAT Mean',
    'bwd_iat_std': ' Bwd IAT Std',
    'bwd_iat_max': ' Bwd IAT Max',
    'bwd_iat_min': ' Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': ' Bwd PSH Flags',
    'fwd_urg_flags': ' Fwd URG Flags',
    'bwd_urg_flags': ' Bwd URG Flags',
    'fwd_header_len': ' Fwd Header Length',
    'bwd_header_len': ' Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': ' Bwd Packets/s',
    'pkt_len_min': ' Min Packet Length',
    'pkt_len_max': ' Max Packet Length',
    'pkt_len_mean': ' Packet Length Mean',
    'pkt_len_std': ' Packet Length Std',
    'pkt_len_var': ' Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'syn_flag_cnt': ' SYN Flag Count',
    'rst_flag_cnt': ' RST Flag Count',
    'psh_flag_cnt': ' PSH Flag Count',
    'ack_flag_cnt': ' ACK Flag Count',
    'urg_flag_cnt': ' URG Flag Count',
    'cwe_flag_count': ' CWE Flag Count',
    'ece_flag_cnt': ' ECE Flag Count',
    'down_up_ratio': ' Down/Up Ratio',
    'pkt_size_avg': ' Average Packet Size',
    'fwd_seg_size_avg': ' Avg Fwd Segment Size',
    'bwd_seg_size_avg': ' Avg Bwd Segment Size',
    'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
    'fwd_pkts_b_avg': ' Fwd Avg Packets/Bulk',
    'fwd_blk_rate_avg': ' Fwd Avg Bulk Rate',
    'bwd_byts_b_avg': ' Bwd Avg Bytes/Bulk',
    'bwd_pkts_b_avg': ' Bwd Avg Packets/Bulk',
    'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
    'subflow_fwd_pkts': 'Subflow Fwd Packets',
    'subflow_fwd_byts': ' Subflow Fwd Bytes',
    'subflow_bwd_pkts': ' Subflow Bwd Packets',
    'subflow_bwd_byts': ' Subflow Bwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': ' Init_Win_bytes_backward',
    'fwd_act_data_pkts': ' act_data_pkt_fwd',
    'fwd_seg_size_min': ' min_seg_size_forward',
    'active_mean': 'Active Mean',
    'active_std': ' Active Std',
    'active_max': ' Active Max',
    'active_min': ' Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': ' Idle Std',
    'idle_max': ' Idle Max',
    'idle_min': ' Idle Min'
}

    while True:
        novos_dados = pd.read_csv('Syc4.csv', low_memory=False)

        colunas_remover = [0, 1, 4, 5, 7]
        data_filtrada = novos_dados.drop(columns=novos_dados.columns[colunas_remover])

        # Calcula a matriz de correlação
        data_filtrada = data_filtrada.select_dtypes(include=['float64', 'int64'])
        data_filtrada = pd.get_dummies(data_filtrada)

        #Retirar linhas dos valores faltantes
        data_filtrada = data_filtrada.dropna()

        # Substitua os valores infinitos
        data_filtrada = data_filtrada.replace([np.inf, -np.inf], 3.4028235e+38)



        # Trocar a posição das colunas
        colunas_master = ['src_port', 'dst_port', 'protocol', 'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len', 'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']
        # Filtrar apenas as colunas presentes no DataFrame original
        #colunas_master = [coluna for coluna in colunas_master if coluna in data_filtrada.columns]
        data_filtrada = data_filtrada[colunas_master]

        # Renomear as colunas de acordo com o mapeamento
        data_filtrada.rename(columns=mapeamento_colunas, inplace=True)

        # Escalonamento dos atributos
        scaler_data_filtrada = StandardScaler()
        data_filtrada = scaler_data_filtrada.fit_transform(data_filtrada)


        #print(data_filtrada.columns)
        #exit()
    
        #Chamando função de realizar predict
        previsoes_floresta = analisar_trafego(data_filtrada, modelo_floresta_aleatoria)
            #previsoes_arvore = analisar_trafego(data_filtrada, modelo_arvore_decisao)
    
        for resultado in previsoes_floresta:
            tomar_acoes(resultado)
        #for resultado in previsoes_arvore:
        #    tomar_acoes(resultado)
            
        time.sleep(5)
