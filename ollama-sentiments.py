import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import ollama
from datetime import datetime

def normalize(string):
  try:
      value = float(string)
      return value
  except ValueError:
      return 0

def textTime(start, end):
  return f"\t{(end-start):.03f}s"

def sentiment_analysis(model_name, text, index=0):
  start = time.time()
  # Mensagem que será enviada para o modelo

  # content = f'''
  #   Você é um especialista em análise de sentimentos. Analise o texto entre colchetes [] e forneça uma nota entre -1.00 e 1.00 para o sentimento expressado no texto, sendo -1.00 muito ruim e 1.00 muito bom.
  #   Contexto dos textos: avaliações de aplicativo de banco.
  #   Seja o mais criterioso e preciso possível: o modelo é muito sensível e pequenas diferenças podem resultar em notas diferentes.
  #   Ao analisar o sentimento, considere os seguintes critérios:
  #   Positividade e Negatividade: Determine se o texto expressa uma opinião positiva ou negativa sobre o aplicativo de banco
  #   Intensidade das Emoções: Avalie a intensidade das emoções expressadas. Palavras ou frases que indicam forte satisfação ou insatisfação devem ser refletidas na nota com maior precisão
  #   Neutralidade: Se o texto é predominantemente factual ou descritivo sem expressar fortes emoções, a nota deve estar mais próxima de 0.00.
  #   Não seja excessivamente positivo ou negativo: Seja objetivo e imparcial ao avaliar o sentimento expresso no texto.
  #   Contexto Específico: Considere o contexto específico de avaliações de aplicativos de banco. Problemas técnicos, facilidade de uso, eficiência e suporte ao cliente são aspectos relevantes que podem influenciar o sentimento.
  #   Retorne somente o número com exatamente duas casas decimais que representa o sentimento e nada mais.
  #   Texto: [{text}]
  # '''
  content = f'''
    Você é um especialista em análise de sentimentos. Analise o texto entre colchetes [ ] e atribua uma nota entre -1.0 e 1.0 que represente o sentimento expressado. Use -1.0 para sentimentos muito negativos e 1.0 para sentimentos muito positivos e 0 para sentimentos neutros.
    Também atribua uma nota entre 0.0 e 1.0 o quão confiante você está perante a nota que está dando para o sentimento.
    Atenção: Utilize apenas uma casa decimal.

    Considere os seguintes aspectos linguísticos, com a porcentagem indicando a importância de cada aspecto para a análise de sentimento:

    Léxico Emocional (25%)
    Identifique palavras que expressam sentimentos:
    Positivo: Palavras como "feliz", "excelente", "fantástico".
    Neutro: Palavras informativas, como "adequado", "mediano".
    Negativo: Palavras como "triste", "horrível", "fracasso".
    
    Contexto Linguístico (20%)
    Considere como o contexto altera o sentimento das palavras:
    Positivo: Uso de ironia positiva ("Claro que estou super feliz com isso!" expressando felicidade).
    Neutro: Ironia sem carga emocional significativa ("Muito típico").
    Negativo: Sarcasmo negativo ("Claro, isso foi tão útil" com intenção negativa).
    
    Intensidade Emocional (15%)
    Avalie a força das emoções no texto:
    Positivo: Intensificadores de emoções positivas ("extremamente feliz").
    Neutro: Emoções de pouca intensidade ("levemente contente").
    Negativo: Intensificadores de emoções negativas ("completamente horrível").
   
    Sintaxe e Estrutura da Frase (10%)
    A construção da frase pode influenciar o sentimento:
    Positivo: Frases que transmitem satisfação ("Foi uma experiência incrível").
    Neutro: Frases descritivas e objetivas ("O evento ocorreu às 18 horas").
    Negativo: Frases que expressam frustração ("Isso foi um desperdício de tempo").
    
    Contexto Cultural e Temporal (10%)
    Considere o impacto cultural e temporal nas expressões:
    Positivo: Termos com conotação positiva em contextos atuais ("épico" significando algo muito bom).
    Neutro: Palavras com significado literal e neutro.
    Negativo: Termos com conotação negativa ou desatualizada.
    
    Uso de Emojis e Emoticons (5%)
    Emojis podem modificar o tom emocional:
    Positivo: Emojis de felicidade ou aprovação (😊, 👍).
    Neutro: Emojis sem carga emocional significativa (🔄, 🔍).
    Negativo: Emojis de tristeza ou desaprovação (😞, 👎).
    
    Marcadores Discursivos (5%)
    Identifique palavras que mudam o tom ou introduzem novas ideias:
    Positivo: Marcadores de emoção positiva ("excelente").
    Neutro: Marcadores de transição ("no entanto").
    Negativo: Marcadores de desapontamento ("infelizmente").
    
    Polaridade e Multiplicidade de Sentimentos (5%)
    Detecte a presença de múltiplos sentimentos no texto:
    Positivo: Sentimentos positivos predominantes em contextos mistos.
    Neutro: Equilíbrio ou ambiguidade emocional.
    Negativo: Sentimentos negativos predominantes.
    
    Referências a Entidades Nomeadas e Relacionamentos (5%)
    O modo como entidades são mencionadas pode alterar o sentimento:
    Positivo: Mencionar entidades de forma elogiosa ("A Apple fez um trabalho excelente").
    Neutro: Referências neutras ou informativas.
    Negativo: Mencionar entidades de forma crítica ("O governo falhou miseravelmente").
    
    Estilo de Linguagem e Tom (5%)
    A tonalidade geral do texto pode impactar a percepção emocional:
    Positivo: Linguagem otimista.
    Neutro: Estilo informativo ou descritivo.
    Negativo: Linguagem pessimista ou crítica.
    
    Objetividade e Neutralidade (5%)
    Avalie se o texto é imparcial ou inclinado:
    Positivo: Fatos com leve inclinação positiva.
    Neutro: Fatos estritamente objetivos.
    Negativo: Fatos com leve inclinação negativa.
    
    Entonação e Prosódia (em Texto Falado) (5%)
    Considere como a entonação pode influenciar o sentimento:
    Positivo: Entonação alegre ou excitada.
    Neutro: Entonação neutra e equilibrada.
    Negativo: Entonação triste ou irritada.

    Formato de Retorno: Apenas numero com uma casa decimal, e o número que representa os sentimentos e o numero que representa o quão confiante está perante a nota do sentimento. por exemplo "0.2 0.9". 
    Não forneça explicações adicionais.
    Caso não haja nenhum texto para ser lido não retorne nada.

    Texto: [{text}]
  '''
  try:
    # Instacia do modelo e envio da mensagem
    response = ollama.chat(
      model_name,
      messages=[{'role': 'user', 'content': content}],
    )

    # Verificação do sentimento retornado
    response_content = response['message']['content'].strip().split()
    sentiment = normalize(response_content[0])
    confidence = normalize(response_content[1])
    end = time.time()

    df.at[index, 'time'] = float(f"{(end-start):.03f}")
    
    prefix = f"{index} : {len(df.index)}"
    print(f"{prefix} | Sentimento: {sentiment} | Confiança: {confidence} | {textTime(start, end)}")

    return sentiment, confidence
  except Exception as e:
      return 0, 0

def plot_histograma(dataFrame):
  # Plotando o gráfico
  # Visualizar a distribuição dos sentimentos
  plt.figure(figsize=(10, 6))
  sns.histplot(dataFrame['sentiment'], bins=30, kde=True, color='purple')
  plt.title('Distribuição dos Sentimentos ')
  plt.xlabel('Sentimento')
  plt.ylabel('Frequência')
  plt.xticks(np.arange(-1.0, 1.0, step=0.20))
  plt.savefig(f'{history_model_save_folder}sentiment_histograma.png')

def plot_heatmap(dataFrame):
   # Arredondar a pontuação compound para facilitar a visualização
  dataFrame['sentiment_rounded'] = dataFrame['sentiment'].round(1)

  # Criar a tabela de contingência
  heatmap_data = pd.crosstab(dataFrame['sentiment_rounded'], dataFrame['score'])

  # Ordenar os valores do sentiment do maior para o menor
  heatmap_data = heatmap_data.sort_index(ascending=False)

  # Calcular a somatória de registros em cada nota do NPS
  nps_totals = heatmap_data.sum(axis=0)

  # Criar labels customizadas para o eixo x
  nps_labels = [f'{nps}\n({total})' for nps, total in nps_totals.items()]

  # Gerar o heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='d')
  plt.title('Heatmap de Comparação entre Nota de NPS e Sentimento')
  plt.xlabel('Nota de NPS')
  plt.ylabel('Sentimento (Arredondada)')
  plt.xticks(ticks=np.arange(len(nps_labels))+0.5, labels=nps_labels, rotation=0)  # Customizar os labels do eixo x
  plt.yticks(rotation=0)  # Para garantir que as labels do eixo y estejam alinhadas corretamente
  plt.savefig(f'{history_model_save_folder}sentiment_heatmap.png')

def createHistoryFolder(folder):
  if not folder in os.listdir("./history"):
    os.mkdir(f"./history/{folder}")

def saveDataFrame():
  df.to_csv(f'{history_model_save_folder}score_sentiment.csv', index=False, encoding='utf-8')
  plot_histograma(df)
  plot_heatmap(df)

entreprise = "ton"
ollama_models = [
  # {
  #   "model_name": "qwen2",
  #   "run": 3,
  #   "refining": False
  # },
  # {
  #   "model_name": "mistral-nemo",
  #   "run": 3,
  #   "refining": False
  # },
  # {
  #   "model_name": "gemma2",
  #   "run": 3,
  #   "refining": False
  # },
  # {
  #   "model_name": "phi3",
  #   "run": 3,
  #   "refining": False
  # },
  {
    "model_name": "deepseek-coder-v2",
    "run": 3,
    "refining": False
  },
  {
    "model_name": "llama3.1",
    "run": 3,
    "refining": False
  },
]

for model in ollama_models:
  for i in range(model['run']):
    df = pd.read_csv(f'score_{entreprise}.csv')
    df['sentiment'] = np.nan
    df['confidence'] = np.nan

    history_model_folder = f"{model['model_name'].replace(":", "_")}"
    createHistoryFolder(history_model_folder)

    history_model_now_folder = f"{history_model_folder}/{entreprise}-{str(datetime.now()).replace(":", "-")}/"
    createHistoryFolder(history_model_now_folder)

    history_model_save_folder = f"./history/{history_model_now_folder}"

    startGlobal = time.time()
    print("Iniciando a analise de sentimentos")

    # Aplicando a analise de sentimentos
    for index, row in df.iterrows():
      if('sentiment' in df.columns):
        df.at[index, 'sentiment'], df.at[index, 'confidence'] = sentiment_analysis(model['model_name'], row['text'], index)
        if index % 10 == 0 and index != 0 and model['refining'] == True:
          saveDataFrame()

    saveDataFrame()
    print(f"Tempo Total: {textTime(startGlobal, time.time())}")

