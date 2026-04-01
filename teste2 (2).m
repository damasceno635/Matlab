clear
clc

% carregar dataset
data = readtable('iris.data','FileType','text');

X = table2array(data(:,1:4));
labels = data{:,5};

% converter classes para numeros
labels = grp2idx(labels);

percentuais = 0.1:0.1:0.8;

resultados = zeros(20,length(percentuais));

for p = 1:length(percentuais)

    perc = percentuais(p);

    for exec = 1:20

        idx = randperm(length(labels));
        Xsh = X(idx,:);
        Ysh = labels(idx);

        ntrain = round(perc*length(labels));

        Xtrain = Xsh(1:ntrain,:);
        Ytrain = Ysh(1:ntrain);

        Xtest = Xsh(ntrain+1:end,:);
        Ytest = Ysh(ntrain+1:end);

        classes = unique(Ytrain);

        centroides = zeros(length(classes),4);

        for c = 1:length(classes)
            centroides(c,:) = mean(Xtrain(Ytrain==classes(c),:));
        end

        acertos = 0;

        for i = 1:size(Xtest,1)

            for c = 1:length(classes)
                dist(c) = norm(Xtest(i,:) - centroides(c,:));
            end

            [~,classe_pred] = min(dist);

            if classe_pred == Ytest(i)
                acertos = acertos + 1;
            end

        end

        resultados(exec,p) = acertos/length(Ytest);

    end
end

minimo = min(resultados);
maximo = max(resultados);
media = mean(resultados);
desvio = std(resultados);

tabela = table(percentuais', minimo', maximo', media', desvio', ...
'VariableNames',{'Treino','Min','Max','Media','Desvio'});

disp(tabela)