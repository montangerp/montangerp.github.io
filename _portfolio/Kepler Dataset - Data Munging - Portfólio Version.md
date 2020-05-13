
This notebook was prepared by <a href="https://montangerp.github.io/" target="_blank"> Patricia Montanger</a>. Source and license info is on <a href="https://github.com/montangerp" target="_blank">GitHub</a>. <br>
This research made use of <a href="https://docs.lightkurve.org/index.html" target="_blank">Lightkurve</a>, a Python package for Kepler and TESS data analysis (Lightkurve Collaboration, 2018).

# Manipulação de dados 

## Descrição

Neste notebook é realizado o processo de limpeza, reestruturação e reformatação de dados, mais especificamente de séries temporais, para que fiquem em um formato adequado para análise.

Os dados utilizados são curvas de luz provenientes de milhares de estrelas e adquiridas através do telescópio **Kepler**, onde cada ponto da curva representa a magnitude da estrela em função do tempo. Queremos obter uma quantidade significativa de curvas, para posteriormente realizar pesquisas para encontrar padrões nas séries temporais e utiliza-las em algoritmos de machine learning, auxiliando assim os astronômos na identificação de exoplanetas.
    
<img src="esolc.jpg" style="width:500px;height:400px"/>

<div style="text-align: center"><a href="https://www.eso.org/public/images/eso1023f/" target="_blank"> Fonte </a>       </div>

A fonte de dados utilizada foi o <a href="https://exoplanetarchive.ipac.caltech.edu/" target="_blank">NASA Exoplanet Archive</a>, um catálogo online que coleta e correlaciona dados e informações astronômicas sobre exoplanetas e suas estrelas hospedeiras, além de fornecer ferramentas para trabalhar com esses dados. Esses dados incluem parâmetros estelares como posições, magnitudes, temperaturas e parâmetros de exoplanetas como massas, parâmetros orbitais e dados de descoberta e caracterização como curvas de velocidade radial, curvas de luz fotométricas, imagens e espectros.

## Objetivo

Nosso objetivo é importar e manipular dados do Kepler para posteriormente aplicar em algoritmos de machine learning.

- Primeiro passo: através da página de arquivos do kepler importamos os nomes dos objetos (Kepler ID), suas respectivas classes e quarters.

- Segundo passo: importar os dados para o Jupyter e explorar os dados.

- Terceiro passo: selecionar apenas as classes de interesse.

- Quarto passo: remover valores ausentes.

- Quinto passo: selecionar curvas de acordo com o quarter desejado pelo usuário.

- Sexto passo: com a biblioteca Lightcurve fazer o download do fluxo das curvas de luz.

- Sétimo passo: remover valores ausentes.

- Oitavo passo: adicionar as classes para os fluxos importados.

- Nono passo: salvar arquivos csv.

- Décimo passo: visualizar resultado final.

### Primeiro passo

Na página do NASA Exoplanet Archive selecionamos apenas as colunas de dados de interesse e realizamos o download do arquivo CSV contendo os parâmetros:
>**KepID<br>
Exoplanet Archive Disposition<br>
Quarters**




<img src="exoarchive.png"  style="width:800px;height:400px"/>

<div style="text-align: center"><a href="https://exoplanetarchive.ipac.caltech.edu/" target="_blank"> Fonte </a>       </div>

### Segundo passo


```python
import pandas as pd
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from lightkurve import search_lightcurvefile
```

Utilizaremos a biblitoeca Lightkurve que nos auxiliará quando desejarmos importar as curvas luz, está biblioteca oferece uma maneira fácil de analisar dados de séries temporais obtidos por telescópios, em particular as das missões Kepler e TESS da NASA.


```python
lc = pd.read_csv("kepler_data_quarter.csv", sep = ",") 
lc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kepid</th>
      <th>koi_disposition</th>
      <th>koi_quarters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10797460</td>
      <td>CONFIRMED</td>
      <td>11111111111111111000000000000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10797460</td>
      <td>CONFIRMED</td>
      <td>11111111111111111000000000000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10811496</td>
      <td>CANDIDATE</td>
      <td>11111101110111011000000000000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10848459</td>
      <td>FALSE POSITIVE</td>
      <td>11111110111011101000000000000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10854555</td>
      <td>CONFIRMED</td>
      <td>01111111111111111000000000000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
lc.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kepid</th>
      <th>koi_disposition</th>
      <th>koi_quarters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9559</th>
      <td>10090151</td>
      <td>FALSE POSITIVE</td>
      <td>11111101110111011000000000000000</td>
    </tr>
    <tr>
      <th>9560</th>
      <td>10128825</td>
      <td>CANDIDATE</td>
      <td>11111111111111111000000000000000</td>
    </tr>
    <tr>
      <th>9561</th>
      <td>10147276</td>
      <td>FALSE POSITIVE</td>
      <td>11111111111111111000000000000000</td>
    </tr>
    <tr>
      <th>9562</th>
      <td>10155286</td>
      <td>CANDIDATE</td>
      <td>11111101110111011000000000000000</td>
    </tr>
    <tr>
      <th>9563</th>
      <td>10156110</td>
      <td>FALSE POSITIVE</td>
      <td>11111101110111011000000000000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# comparação da quantidade de objetos por classe
lc.groupby(['koi_disposition','koi_disposition']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
```


![png](output_13_0.png)



```python
lc.dtypes
```




    kepid               int64
    koi_disposition    object
    koi_quarters       object
    dtype: object




```python
lc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9564 entries, 0 to 9563
    Data columns (total 3 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   kepid            9564 non-null   int64 
     1   koi_disposition  9564 non-null   object
     2   koi_quarters     8422 non-null   object
    dtypes: int64(1), object(2)
    memory usage: 224.3+ KB
    

Observe que temos 3 classes: 
>**Falsos positivos<br>Candidatos<br>Confirmados** <br>

Porém queremos apenas os falsos positivos e confirmados de forma que vamos iniciar retirando todos os itens classificados como candidatos.
Além disso visualizamos acima que em koi_quarters existem valores ausentes, os quais precisarão ser removidos ou substituídos.

### Terceiro passo


```python
lc_array = np.array(lc)
print('total inicial de curvas: %d'%(len(lc_array)))
```

    total inicial de curvas: 9564
    


```python
# retira os labels 'candidate' da base de dados: nova base lc_array_wc 
lc_array_wc = []
for i in range(0, len(lc_array)):
    if lc_array[i][1] == 'CANDIDATE':
        np.delete(lc_array, i, 0)
    else:
        lc_array_wc.append(lc_array[i])
        
# quantidade de itens em cada classe 
confirmed = []
false_positive = []
for i in range(0, len(lc_array_wc)):
    if lc_array_wc[i][1] == 'CONFIRMED':
        confirmed.append(lc_array_wc[i])
    else:
        false_positive.append(lc_array_wc[i])

print('falsos positivos: %d, confirmados: %d\n\ntotal atualizado: %d'%(len(false_positive),len(confirmed), len(false_positive)+len(confirmed)))
```

    falsos positivos: 4841, confirmados: 2303
    
    total atualizado: 7144
    


```python
# comparação da quantidade de objetos por classe
df = pd.DataFrame(data=lc_array_wc, columns=["kepid", "koi_disposition", "koi_quarters"])
df.groupby(['koi_disposition','koi_disposition']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
```


![png](output_21_0.png)


### Quarto passo

Temos agora nossos dados apenas com os labels de interesse e vamos excluír os valores ausentes existentes na coluna quarters.


```python
# exclui as curvas com valores ausentes da coluna 'quarters'
lc_array_wc_wn = []
for i in range(0, len(lc_array_wc)):
    h = lc_array_wc[i][2]
    if pd.isnull(h) == True:
        np.delete(lc_array_wc, i, 0)
    else:
        lc_array_wc_wn.append(lc_array_wc[i])
        
print('total atualizado: %d'%(len(lc_array_wc_wn)))
```

    total atualizado: 6465
    

### Quinto passo

Neste passo temos que entender que para usar a biblioteca Lightkurve precisamos selecionar o quarter (ou trimestre) que desejamos que nossa base de curvas de luz possua. Os trimestres variam entre 1 e 17 e são representados de forma binária:

  kepid | koi_disposition|          koi_quarters
--------|----------------| --------------------------------
10797460|   CONFIRMED    | 11111111111111111000000000000000
10811496|   CONFIRMED    | 11111101110111011000000000000000
10848459| FALSE POSITIVE | 11111110111011101000000000000000

O **número 1** nos diz que existe uma curva de luz dísponivel para aquele quarter, já o **número 0** nos diz que não há uma curva para aquele quarter. Sabendo disso, temos que selecionar o quarter desejado antes de importar as curvas.


```python
quarter = 1           # alterar aqui de acordo com o valor de quarter desejado entre 1 e 17  
```


```python
# seleciona apenas as curvas com o quarter escolhido
lc_array_wc_wn_Q = []
for i in range(0, len(lc_array_wc_wn)):
    if len(lc_array_wc_wn[i][2]) != 32 and len(lc_array_wc_wn[i][2]) != (32 - quarter + 1):
        np.delete(lc_array_wc_wn, i, 0)
    else:
        lc_array_wc_wn_Q.append(lc_array_wc_wn[i]) 
lc_array_wc_wn_Q_sq = []                               
for i in range(0, len(lc_array_wc_wn_Q)):
    if len(lc_array_wc_wn_Q[i][2]) == 32 and lc_array_wc_wn_Q[i][2][quarter-1] == '0':
        np.delete(lc_array_wc_wn_Q, i, 0)
    elif len(lc_array_wc_wn_Q[i][2]) == (32 - quarter + 1) and lc_array_wc_wn_Q[i][2][0] == '0':
        np.delete(lc_array_wc_wn_Q, i, 0)
    else:
        lc_array_wc_wn_Q_sq.append(lc_array_wc_wn_Q[i]) 

print('Quarter = %d \n\nTotal atualizado: %d'%((quarter),len(lc_array_wc_wn_Q_sq)))
```

    Quarter = 1 
    
    Total atualizado: 5297
    

Na nova lista estão apenas as curvas que possuem o **quarter = 1**, portanto temos agora um número atualizado no total de curvas.


```python
# quantidade de itens e porcentagem em cada classe para o quarter selecionado
confirmed_ = []
false_positive_ = []
for i in range(0, len(lc_array_wc_wn_Q_sq)):
    if lc_array_wc_wn_Q[i][1] == 'CONFIRMED':
        confirmed_.append(lc_array_wc_wn_Q_sq[i])
    else:
        false_positive_.append(lc_array_wc_wn_Q_sq[i])

total_lc = len(lc_array_wc_wn_Q_sq)
porcent_fp = (100*len(false_positive_))/total_lc
porcent_c = (100*len(confirmed_))/total_lc 

print('falsos positivos: %d \t\t %f porcento \nconfirmados: %d \t\t %f porcento \n\ntotal atualizado: %d'%(len(false_positive_),porcent_fp,len(confirmed_),porcent_c,len(false_positive_)+len(confirmed_)))
```

    falsos positivos: 3007 		 56.767982 porcento 
    confirmados: 2290 		 43.232018 porcento 
    
    total atualizado: 5297
    


```python
# comparação da quantidade de objetos por classe
dfq = pd.DataFrame(data=lc_array_wc_wn_Q, columns=["kepid", "koi_disposition", "koi_quarters"])
dfq.groupby(['koi_disposition','koi_disposition']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
```


![png](output_31_0.png)


### Sexto passo

Agora vamos importar as curvas através do Lightcurve, neste exemplo importamos apenas algumas das curvas disponíveis, é possível importar todas porém é um processo que levaria horas e exigiria muito de um computador comum, **é recomendável utilizar um GPU/Cluster para executar esta etapa** caso queira trabalhar com todas as curvas.


```python
# importando as curvas de luz com o KeplerGO
lc_flux = []
start_time = time.time()
for i in range(0,10):                                           # durante testes altere a quantidade de curvas aqui
#for i in range(0, len(lc_array_wc_wn_Q_sq)):                   # descomente esta linha para importar todas as curvas
    ep = str(lc_array_wc_wn_Q_sq[i][0])                                                     # seleciona os KeplerID
    lcf = search_lightcurvefile(ep, quarter = quarter).download()         
    lcf = lcf.PDCSAP_FLUX                                             
    lc_flux.append(lcf.flux)
t = time.time() - start_time   

print('\nTempo para importar curvas de luz: %f seconds\n' %t)
```

    gzip was not found on your system! You should solve this issue for astroquery.eso to be at its best!
    On POSIX system: make sure gzip is installed and in your path!On Windows: same for 7-zip (http://www.7-zip.org)!
    
    Tempo para importar curvas de luz: 17.964212 seconds
    
    


```python
# visualização das curvas importadas
fig = plt.figure(figsize=(10, 10)) 
fig_dims = (3, 2)
dflux = pd.DataFrame(data=lc_flux)
plt.subplot2grid(fig_dims, (0, 0)),plt.plot(dflux.iloc[0],'.')
plt.subplot2grid(fig_dims, (0, 1)),plt.plot(dflux.iloc[2],'.')
plt.subplot2grid(fig_dims, (1, 0)),plt.plot(dflux.iloc[4],'.')
plt.subplot2grid(fig_dims, (1, 1)),plt.plot(dflux.iloc[9],'.')
plt.show()
```


![png](output_35_0.png)


Já com as curvas na lista *lc_flux* vamos verificar se todas as curvas importadas possuem o mesmo tamanho e apresentar uma mensagem caso alguma tenha um tamanho diferente, raramente existem curvas de tamanhos diferentes para o mesmo quarter, porém sempre realizaremos esta verificação. 


```python
# verificar se todas as curvas importadas possuem o mesmo tamanho
for i in range(0,len(lc_flux)):
    if len(lc_flux[i]) != len(lc_flux[i-1]):
        print("A curva %d possui tamanho diferente das demais curvas. Tamanho: %d "%(i,len(lc_flux[i])))
print("Caso nenhuma das curvas apresente tamanho diferente, todas as curvas possuem o total de %d pontos cada.\n"%len(lc_flux[3]))
```

    Caso nenhuma das curvas apresente tamanho diferente, todas as curvas possuem o total de 1626 pontos cada.
    
    

### Sétimo passo

Assim como na tabela inicial existiam valores ausentes, nas curvas recém importadas também existem pontos ausentes então vamos identificar estes pontos a seguir e guardar seus indices para verificar a procentagem de pontos que precisaram ser substituídos e assim garantir que não sejam suficientes grandes para alterar as análises feitas no futuro utilizando esta base de dados. Esses dados serão salvos em uma tabela CSV caso seja necessário identificá-los em trabalhos futuros.


```python
# percorre todos os pontos de todas as curvas verificando onde existem NaN, guarda os indices e quantidades
count = 0
notanumber = []
notanumber.append(('Ponto da curva','Curva'))
for j in range(0, len(lc_flux)):
    for i in range(0,len(lc_flux[j])):
        h = lc_flux[j][i]
        if pd.isnull(h) == True:
            notanumber.append((i,j))
            count+=1
 
# salva tabela csv com curvas e pontos contendo nan 
csvfile_nan = "nan_Q1.csv"
with open(csvfile_nan, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(notanumber)
            
print("Total de pontos com valores ausentes: %d\n" %count)
```

    Total de pontos com valores ausentes: 20
    
    

Para lidar com os valores ausentes escolhemos uma função que os substitui automaticamente por zero.


```python
# porcentagem de dados que serão subsituidos por zero 
total = len(lc_flux)*len(lc_flux[0])
porcent = (100*count)/total

print("Porcentagem de valores ausentes substituídos por zero: %f porcento\n" %porcent)

# funçao que substitui NaN por zero e Inf por números finitos
lc_flux_ = np.nan_to_num(lc_flux)
```

    Porcentagem de valores ausentes substituídos por zero: 0.123001 porcento
    
    

Observamos que a porcentagem de valores ausentes não é grande o suficiente para interferir nas análises, então vamos para a próxima etapa.

### Oitavo passo

Vamos adicionar as classes correspondentes as curvas que importamos.


```python
# aqui adicionamos os labels ao lc_flux (curvas de luz) 
# labels estão no lc_array_wc_wn_Q1_sq
selected_list = []
for i in range(0,len(lc_flux_)):
    selected_list.append(lc_flux_[i].tolist())
    selected_list[i].append(lc_array_wc_wn_Q_sq[i][1])

selected_list = list(map(list, zip(*selected_list)))

label = "Label"
selected_list [len(lc_flux_[3])].insert(0,label)     

flux = "Flux"
for i in range(0,len(lc_flux_[3])):
    selected_list[i].insert(0,flux)
    
selected_list = list(map(list, zip(*selected_list)))
```

Se houver valores ausentes restantes na base de dados, nossa análise será prejudicada, então por redundância vamos realizar a verificação novamente.


```python
# redundância para verificar se existem NaN
# Obs.: deve sempre ser zero
count_ = 0
notanumber_ = []
notanumber_.append(('Ponto da curva','Curva'))
for j in range(0, len(selected_list)):
    for i in range(0,len(selected_list[j])):
        h = selected_list[j][i]
        if pd.isnull(h) == True:
            notanumber_.append((i,j))
            count_+=1

print("Total de pontos contendo valores ausentes (segunda verificação): %d  ----> deve sempre ser zero" %count_)
```

    Total de pontos contendo valores ausentes (segunda verificação): 0  ----> deve sempre ser zero
    

### Nono passo

Finalmente temos a base de dados pronta para análise e podemos salvá-la em um arquivo CSV.


```python
# salvando tabela final em formato csv. já com quarter, labels e fluxo definidos
csvfile = "dados_Q1.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(selected_list)
```

### Décimo passo


```python
pd.read_csv("dados_Q1.csv", sep = ",").head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flux</th>
      <th>Flux.1</th>
      <th>Flux.2</th>
      <th>Flux.3</th>
      <th>Flux.4</th>
      <th>Flux.5</th>
      <th>Flux.6</th>
      <th>Flux.7</th>
      <th>Flux.8</th>
      <th>Flux.9</th>
      <th>...</th>
      <th>Flux.1617</th>
      <th>Flux.1618</th>
      <th>Flux.1619</th>
      <th>Flux.1620</th>
      <th>Flux.1621</th>
      <th>Flux.1622</th>
      <th>Flux.1623</th>
      <th>Flux.1624</th>
      <th>Flux.1625</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11288.066406</td>
      <td>11292.760742</td>
      <td>11281.872070</td>
      <td>11298.019531</td>
      <td>11284.711914</td>
      <td>11292.368164</td>
      <td>11296.479492</td>
      <td>11288.382812</td>
      <td>11288.015625</td>
      <td>11302.500977</td>
      <td>...</td>
      <td>11262.921875</td>
      <td>11250.892578</td>
      <td>11251.499023</td>
      <td>11249.472656</td>
      <td>11255.666016</td>
      <td>11259.477539</td>
      <td>11254.582031</td>
      <td>11268.262695</td>
      <td>11263.854492</td>
      <td>CONFIRMED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11288.066406</td>
      <td>11292.760742</td>
      <td>11281.872070</td>
      <td>11298.019531</td>
      <td>11284.711914</td>
      <td>11292.368164</td>
      <td>11296.479492</td>
      <td>11288.382812</td>
      <td>11288.015625</td>
      <td>11302.500977</td>
      <td>...</td>
      <td>11262.921875</td>
      <td>11250.892578</td>
      <td>11251.499023</td>
      <td>11249.472656</td>
      <td>11255.666016</td>
      <td>11259.477539</td>
      <td>11254.582031</td>
      <td>11268.262695</td>
      <td>11263.854492</td>
      <td>CONFIRMED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8199.683594</td>
      <td>8203.690430</td>
      <td>8201.837891</td>
      <td>8202.610352</td>
      <td>8200.041016</td>
      <td>8194.414062</td>
      <td>8203.027344</td>
      <td>8203.117188</td>
      <td>8204.064453</td>
      <td>8191.709473</td>
      <td>...</td>
      <td>8198.636719</td>
      <td>8190.932129</td>
      <td>8196.329102</td>
      <td>8201.858398</td>
      <td>8191.613281</td>
      <td>8196.993164</td>
      <td>8197.906250</td>
      <td>8199.167969</td>
      <td>8198.675781</td>
      <td>FALSE POSITIVE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>124407.843750</td>
      <td>124410.914062</td>
      <td>124413.546875</td>
      <td>124393.281250</td>
      <td>124406.789062</td>
      <td>124403.601562</td>
      <td>124421.164062</td>
      <td>124411.929688</td>
      <td>124396.515625</td>
      <td>124418.281250</td>
      <td>...</td>
      <td>124415.921875</td>
      <td>124399.828125</td>
      <td>124420.140625</td>
      <td>124417.203125</td>
      <td>124409.242188</td>
      <td>124407.023438</td>
      <td>124386.234375</td>
      <td>124405.234375</td>
      <td>124400.570312</td>
      <td>FALSE POSITIVE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6559.518555</td>
      <td>6552.599609</td>
      <td>6548.751953</td>
      <td>6554.393066</td>
      <td>6552.321289</td>
      <td>6561.672852</td>
      <td>6553.577637</td>
      <td>6556.897461</td>
      <td>6554.946289</td>
      <td>6552.674316</td>
      <td>...</td>
      <td>6535.249023</td>
      <td>6532.930176</td>
      <td>6536.879883</td>
      <td>6532.363281</td>
      <td>6539.292480</td>
      <td>6529.866211</td>
      <td>6531.430176</td>
      <td>6520.676270</td>
      <td>6516.936035</td>
      <td>CONFIRMED</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1627 columns</p>
</div>




```python
pd.read_csv("nan_Q1.csv", sep = ",").head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ponto da curva</th>
      <th>Curva</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>979</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>979</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1167</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>979</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


