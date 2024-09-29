from flask import Flask, jsonify, request, render_template, abort
#import joblib
import numpy as np
import pickle
#import os

# Inicializar o app Flask
app = Flask(__name__)

# Carregar os modelos (certifique-se de que os caminhos estão corretos)
loaded_model = pickle.load(open(r'modelo_regressao.pkl', 'rb'))
#loaded_model = pickle.load(open(r'D:\Projeto\DS\modelo_regressao.pkl', 'rb'))
#knn_r_model = joblib.load(r'D:\Projeto\DS\modelo_knn_regressao.pkl')

# Mapeamentos para as opções categóricas
ufs = ['SP', 'MG', 'SC', 'PR', 'RJ']
escolaridades = ['Superior Cursando', 'Superior Completo', 'Segundo Grau Completo']
estados_civis = ['Solteiro', 'Casado', 'Divorciado']
faixas_etarias = ['18-25', '26-35', '36-45', '46-60', 'Acima de 60']

uf_map = {ufs[i]: i for i in range(len(ufs))}
escolaridade_map = {escolaridades[i]: i for i in range(len(escolaridades))}
estado_civil_map = {estados_civis[i]: i for i in range(len(estados_civis))}
faixa_etaria_map = {faixas_etarias[i]: i for i in range(len(faixas_etarias))}

# Página principal
@app.route('/')
def home():
    return '''
    <h1>Prever Score e Rating de Crédito</h1>
    <form action="/predict" method="post">
        <label>UF:</label><input type="text" name="UF" required><br>
        <label>Escolaridade:</label><input type="text" name="ESCOLARIDADE" required><br>
        <label>Estado Civil:</label><input type="text" name="ESTADO_CIVIL" required><br>
        <label>Quantidade de Filhos:</label><input type="number" name="QT_FILHOS" required><br>
        <label>Possui Casa Própria (Sim/Não):</label><input type="text" name="CASA_PROPRIA" required><br>
        <label>Quantidade de Imóveis:</label><input type="number" name="QT_IMOVEIS" required><br>
        <label>Valor dos Imóveis:</label><input type="number" name="VL_IMOVEIS" required><br>
        <label>Possui outra renda (Sim/Não):</label><input type="text" name="OUTRA_RENDA" required><br>
        <label>Valor da Outra Renda:</label><input type="number" name="OUTRA_RENDA_VALOR"><br>
        <label>Tempo Último Emprego (meses):</label><input type="number" name="TEMPO_ULTIMO_EMPREGO_MESES" required><br>
        <label>Está trabalhando atualmente (Sim/Não):</label><input type="text" name="TRABALHANDO_ATUALMENTE" required><br>
        <label>Último Salário (R$):</label><input type="number" name="ULTIMO_SALARIO" required><br>
        <label>Quantidade de Carros:</label><input type="number" name="QT_CARROS" required><br>
        <label>Valor Tabela dos Carros (R$):</label><input type="number" name="VALOR_TABELA_CARROS" required><br>
        <label>Faixa Etária:</label><input type="text" name="FAIXA_ETARIA" required><br>
        <button type="submit">Prever</button>
    </form>
    '''

# Endpoint para prever score e rating de crédito
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Extrair os dados recebidos
        UF = uf_map[data['UF']]
        ESCOLARIDADE = escolaridade_map[data['ESCOLARIDADE']]
        ESTADO_CIVIL = estado_civil_map[data['ESTADO_CIVIL']]
        QT_FILHOS = int(data['QT_FILHOS'])
        CASA_PROPRIA = 1 if data['CASA_PROPRIA'].strip().lower() == 'sim' else 0
        QT_IMOVEIS = int(data['QT_IMOVEIS'])
        VL_IMOVEIS = float(data['VL_IMOVEIS'])
        OUTRA_RENDA = 1 if data['OUTRA_RENDA'].strip().lower() == 'sim' else 0
        OUTRA_RENDA_VALOR = float(data['OUTRA_RENDA_VALOR']) if data['OUTRA_RENDA'].strip().lower() == 'sim' else 0
        TEMPO_ULTIMO_EMPREGO_MESES = int(data['TEMPO_ULTIMO_EMPREGO_MESES'])
        TRABALHANDO_ATUALMENTE = 1 if data['TRABALHANDO_ATUALMENTE'].strip().lower() == 'sim' else 0
        ULTIMO_SALARIO = float(data['ULTIMO_SALARIO'])
        QT_CARROS = int(data['QT_CARROS'])
        VALOR_TABELA_CARROS = float(data['VALOR_TABELA_CARROS'])
        FAIXA_ETARIA = faixa_etaria_map[data['FAIXA_ETARIA']]

        # Montar os novos dados
        novos_dados = [
            UF, ESCOLARIDADE, ESTADO_CIVIL, QT_FILHOS, CASA_PROPRIA, QT_IMOVEIS, VL_IMOVEIS,
            OUTRA_RENDA, OUTRA_RENDA_VALOR, TEMPO_ULTIMO_EMPREGO_MESES,
            TRABALHANDO_ATUALMENTE, ULTIMO_SALARIO, QT_CARROS, VALOR_TABELA_CARROS, FAIXA_ETARIA
        ]

        # Transformar os dados para o formato adequado
        X = np.array(novos_dados).reshape(1, -1)

        # Fazer previsões
        score_previsto = loaded_model.predict(X)[0]

        # Verificar status de aprovação
        status = 'Aprovado' if score_previsto > 65 else 'Recusado'
        
        return jsonify({
            'score_previsto': score_previsto,
            'status': status
        })
        
        # Return response with styling
        return f"""
        <h2>Resultado da Previsão:</h2>
        <p><strong style="font-size: 24px;">Score Previsto: {score_previsto:.2f}</strong></p>
        <p><strong style="font-size: 24px;">Status: {status}</strong></p>
        """

        #return jsonify({
        #    'score_previsto': score_previsto,
        #    'status': status
        #})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Rodar o servidor Flask
    app.run(port=5000)

