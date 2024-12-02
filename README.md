# Reconhecimento de Atividades Humanas com K-means

### Objetivo do Projeto
O objetivo deste projeto foi implementar e avaliar o algoritmo K-means para o agrupamento de atividades humanas com base em dados coletados por sensores de smartphones. Utilizamos o dataset "Human Activity Recognition Using Smartphones" da UCI Machine Learning Repository, que contém informações sobre as atividades de 30 participantes realizadas com o auxílio de acelerômetro e giroscópio integrados a um smartphone.

### O projeto se concentrou nas seguintes etapas:

- Análise Exploratória dos Dados: Análise das distribuições, correlações e redução de dimensionalidade usando PCA.
- Implementação do K-means: Aplicação do algoritmo K-means para agrupar as atividades com base nos dados dos sensores.
- Escolha do Número de Clusters (K): Utilização do método do cotovelo e do silhouette score para determinar o número ideal de clusters.
- Avaliação e Visualização dos Resultados: Avaliação da qualidade dos clusters formados e visualização dos mesmos usando PCA.

## Instruções para Executar o Código
- Pré-requisito:   
Certifique-se de ter o Python instalado, é necessário instalar as bibliotecas listadas abaixo, que podem ser instaladas via pip:    
```pip install numpy pandas matplotlib seaborn scikit-learn```

## Passos para Execução

O dataset utilizado está disponível no repositório do GitHub. É possível acessar diretamente a URL raw dos arquivos ou clonar o repositório com o seguinte comando:    
```git clone https://github.com/brendatrindade/Reconhecimento-de-atividade-humana.git```    
    
Após clonar o repositório, acesse a pasta onde o código está localizado. Em seguida, execute o código Python ```reconhecimento.py``` para carregar e processar os dados, aplicar o K-means e gerar as visualizações.
        
O código irá gerar os gráficos de visualização dos clusters, além de calcular as métricas de avaliação como o silhouette score e inércia.

## Resultados: 
O script gerará:   

- Gráficos de clusters em 2D
- As métricas de avaliação de qualidade do modelo, como o silhouette score e a inércia.

### Conclusão e Considerações sobre os Resultados Obtidos

- Desempenho do K-means: O algoritmo K-means demonstrou boa capacidade de segmentar as atividades de forma agrupada. O número ideal de clusters foi escolhido como 6, correspondendo ao número de atividades no dataset.
- Silhouette Score: O silhouette score para K=6 foi alto, o que indica uma boa separação entre os clusters, com cada cluster representando uma atividade distinta de forma razoavelmente bem separada.
- Inércia: A inércia observada foi consistente com o esperado, diminuindo conforme o número de clusters aumentava, até atingir um ponto de estabilização no método do cotovelo.

### Considerações sobre os Resultados

Apesar da boa separação observada entre a maioria dos clusters, houve alguma sobreposição entre as atividades "sentar" e "ficar em pé", indicando que essas atividades têm características similares, dificultando a separação clara.    
O modelo K-means é sensível à inicialização dos centroides e à presença de outliers, o que pode afetar o desempenho. No entanto, a escolha de K-means++ ajudou a melhorar a inicialização dos centroides, garantindo melhor convergência.    
A redução de dimensionalidade através do PCA foi útil para a visualização, mas também pode ter levado à perda de informações relevantes. Em projetos futuros, poderia ser interessante explorar técnicas de redução de dimensionalidade alternativas ou usar modelos supervisionados.
