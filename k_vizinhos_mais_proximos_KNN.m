%% Classificação Iris com k-NN (k=3 e k=5)
% Variando percentual de treinamento de 10% a 80%
% Executa 20 repetições para cada combinação

clear; clc; close all;

%% 1. Carregar os dados - MÉTODO SIMPLES
% Tenta ler o arquivo linha por linha
fid = fopen('iris.data', 'r');
if fid == -1
    error('Arquivo iris.data não encontrado! Verifique se ele está na pasta correta.');
end

% Inicializar arrays
X = [];
y = [];

% Ler linha por linha
linha = fgetl(fid);
while ischar(linha)
    % Dividir a linha por vírgulas
    partes = strsplit(linha, ',');
    
    % Verificar se tem 5 partes (4 atributos + 1 classe)
    if length(partes) == 5
        % Extrair os 4 atributos (convertendo para número)
        atributos = str2double(partes(1:4));
        
        % Extrair a classe
        classe_nome = partes{5};
        
        % Converter classe para número
        if strcmp(classe_nome, 'Iris-setosa')
            classe_num = 1;
        elseif strcmp(classe_nome, 'Iris-versicolor')
            classe_num = 2;
        elseif strcmp(classe_nome, 'Iris-virginica')
            classe_num = 3;
        else
            classe_num = NaN; % Caso não reconheça
        end
        
        % Adicionar aos arrays
        X = [X; atributos];
        y = [y; classe_num];
    end
    
    linha = fgetl(fid);
end

fclose(fid);

% Verificar se carregou corretamente
fprintf('Dados carregados: %d amostras, %d atributos\n', size(X, 1), size(X, 2));
fprintf('Classes encontradas: %d\n', length(unique(y)));

% Se não carregou nenhum dado, dar erro
if size(X, 1) == 0
    error('Nenhum dado foi carregado. Verifique o formato do arquivo iris.data');
end

num_amostras = size(X, 1); % Deve ser 150

% Percentuais de treinamento
perc_treino = 0.1:0.1:0.8; % 10% a 80%
k_valores = [3, 5];
num_repeticoes = 20;

% Estrutura para armazenar resultados
resultados = struct();

%% 2. Loop sobre percentuais e valores de k
for i_k = 1:length(k_valores)
    k = k_valores(i_k);
    acuracias = zeros(length(perc_treino), num_repeticoes);
    
    for i_perc = 1:length(perc_treino)
        p = perc_treino(i_perc);
        n_treino = round(p * num_amostras);
        
        % Garantir valores válidos
        if n_treino < 1
            n_treino = 1;
        end
        if n_treino >= num_amostras
            n_treino = num_amostras - 1;
        end
        
        for r = 1:num_repeticoes
            % Embaralhar índices
            idx_rand = randperm(num_amostras);
            
            % Dividir treino e teste
            idx_treino = idx_rand(1:n_treino);
            idx_teste = idx_rand(n_treino+1:end);
            
            % Verificar se tem dados de teste
            if isempty(idx_teste)
                acuracias(i_perc, r) = NaN;
                continue;
            end
            
            % Separar dados
            X_treino = X(idx_treino, :);
            y_treino = y(idx_treino);
            X_teste = X(idx_teste, :);
            y_teste = y(idx_teste);
            
            % Classificar
            y_pred = knn_classify(X_treino, y_treino, X_teste, k);
            
            % Calcular acurácia
            acertos = sum(y_pred == y_teste);
            acuracias(i_perc, r) = acertos / length(y_teste) * 100;
        end
    end
    
    % Armazenar
    resultados(i_k).k = k;
    resultados(i_k).acuracias = acuracias;
end

%% 3. Mostrar resultados
fprintf('\n========================================\n');
fprintf('RESULTADOS k-NN - DATASET IRIS\n');
fprintf('========================================\n\n');

for i_k = 1:length(k_valores)
    k = resultados(i_k).k;
    acuracias = resultados(i_k).acuracias;
    
    fprintf('>>> k = %d <<<\n', k);
    fprintf('--------------------------------------------------\n');
    fprintf('%% Treino |   Min   |   Max   |  Média  | Desvio Padrão\n');
    fprintf('--------------------------------------------------\n');
    
    for i_perc = 1:length(perc_treino)
        p = perc_treino(i_perc) * 100;
        dados = acuracias(i_perc, :);
        
        % Remover NaN
        dados = dados(~isnan(dados));
        
        if length(dados) > 0
            media = mean(dados);
            desvio = std(dados);
            minimo = min(dados);
            maximo = max(dados);
            
            fprintf('  %3.0f%%    |  %5.2f  |  %5.2f  |  %5.2f  |     %5.2f\n', ...
                    p, minimo, maximo, media, desvio);
        else
            fprintf('  %3.0f%%    |   N/A   |   N/A   |   N/A   |      N/A\n', p);
        end
    end
    fprintf('\n');
end

%% 4. Gerar gráficos
figure('Position', [100, 100, 800, 400]);

for i_k = 1:length(k_valores)
    subplot(1, 2, i_k);
    acuracias = resultados(i_k).acuracias;
    
    % Calcular médias e desvios
    medias = mean(acuracias, 2, 'omitnan');
    desvios = std(acuracias, 0, 2, 'omitnan');
    
    % Plotar
    x_vals = perc_treino * 100;
    errorbar(x_vals, medias, desvios, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
    
    xlabel('Percentual de Treinamento (%)', 'FontSize', 12);
    ylabel('Acurácia Média (%)', 'FontSize', 12);
    title(sprintf('k-NN (k = %d)', resultados(i_k).k), 'FontSize', 14);
    grid on;
    xlim([5 85]);
    ylim([50 100]);
end

sgtitle('Desempenho do k-NN no Dataset Iris - 20 execuções', 'FontSize', 14);

%% 5. Função k-NN
function y_pred = knn_classify(X_treino, y_treino, X_teste, k)
    n_teste = size(X_teste, 1);
    n_treino = size(X_treino, 1);
    
    % Ajustar k se necessário
    if k > n_treino
        k = n_treino;
    end
    
    y_pred = zeros(n_teste, 1);
    
    for i = 1:n_teste
        % Calcular distâncias Euclidianas
        distancias = zeros(n_treino, 1);
        for j = 1:n_treino
            distancias(j) = sqrt(sum((X_treino(j,:) - X_teste(i,:)).^2));
        end
        
        % Encontrar os k menores distâncias
        [~, idx_ordenados] = sort(distancias);
        vizinhos = y_treino(idx_ordenados(1:k));
        
        % Votação
        y_pred(i) = mode(vizinhos);
    end
end