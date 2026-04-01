% =========================================================
% Classificador: Vizinho Mais Próximo (1-NN)
% Dataset: Iris (UCI) — lendo do arquivo iris.data
% Varia treinamento de 10% a 80%, executa 20 vezes cada
% =========================================================

clear; clc;

% --- 1. LER O ARQUIVO iris.data ---
fid = fopen('iris.data', 'r');

X = zeros(150, 4);
Y = zeros(150, 1);

for i = 1:150
    linha = fgetl(fid);
    partes = strsplit(linha, ',');              % separa pelo caractere ','

    X(i, 1) = str2double(partes{1});
    X(i, 2) = str2double(partes{2});
    X(i, 3) = str2double(partes{3});
    X(i, 4) = str2double(partes{4});

    % Converte o nome da classe para número
    classe = strtrim(partes{5});
    if strcmp(classe, 'Iris-setosa')
        Y(i) = 1;
    elseif strcmp(classe, 'Iris-versicolor')
        Y(i) = 2;
    else
        Y(i) = 3;   % Iris-virginica
    end
end

fclose(fid);

fprintf('Dados carregados: %d amostras, %d features\n', size(X,1), size(X,2));

% --- 2. CONFIGURAÇÕES ---
percentuais    = 0.10:0.10:0.80;
num_execucoes  = 20;

fprintf('\n%-10s %-10s %-10s %-10s %-10s\n', ...
    'Treino%', 'Min%', 'Max%', 'Media%', 'DesvPad%');
fprintf('%s\n', repmat('-', 1, 52));

% --- 3. LOOP PELOS PERCENTUAIS ---
for p = percentuais
    acertos = zeros(1, num_execucoes);

    for exec = 1:num_execucoes

        % Embaralha e divide os dados
        idx      = randperm(150);
        n_treino = round(p * 150);

        idx_treino = idx(1:n_treino);
        idx_teste  = idx(n_treino+1:end);

        X_treino = X(idx_treino, :);
        Y_treino = Y(idx_treino);
        X_teste  = X(idx_teste,  :);
        Y_teste  = Y(idx_teste);

        % Classificar cada ponto de teste
        n_teste = size(X_teste, 1);
        Y_pred  = zeros(n_teste, 1);

        for i = 1:n_teste
            diffs        = X_treino - X_teste(i, :);
            dists        = sqrt(sum(diffs .^ 2, 2));
            [~, idx_min] = min(dists);
            Y_pred(i)    = Y_treino(idx_min);
        end

        acertos(exec) = sum(Y_pred == Y_teste) / n_teste * 100;
    end

    % Estatísticas das 20 execuções
    fprintf('%-10.0f %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
        p*100, min(acertos), max(acertos), mean(acertos), std(acertos));
end