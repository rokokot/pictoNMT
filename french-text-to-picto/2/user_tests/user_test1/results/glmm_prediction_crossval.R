#https://cran.r-project.org/web/packages/glmmTMB/glmmTMB.pdf

library(readr) #read_tsv()
library(psych)
library(dplyr)
library(ggplot2)

library(glmmTMB) #mod <- glmmTMB()
library(lme4) #mod <- lme4::glmer()

library(DHARMa) #plot(DHARMa::simulateResiduals()) #https://cran.r-project.org/web/packages/DHARMa/vignettes/DHARMa.html
library(performance) #performance::check_collinearity() -> multicolinéarité : vif, 95 % intervalle confiance

library(car) #Anova
library(emmeans)
library(effects) #plot(allEffects(res))
library(caret) #confusionMatrix()

library(cvTools) #https://cran.r-project.org/web/packages/cvTools/cvTools.pdf

##test cross-validation meilleurs modèles de prédiction

df <- read_tsv(file = "results_test1.tsv") #773 picto
df2 <- subset(df, select = c(annotator1_picto_all, participant, source_sentence, picto_evaluated, translation, level, V2, V3, V4, V5, V6, V7, V8, sentence_type, pos))
df2 <- na.omit(df2) #683 picto
df_sentence <- df %>% distinct(n_sentence, .keep_all = T) #180 sentences

##picto

set.seed(1234) #pour la reproductibilité
folds <- cvFolds(nrow(df2), K = 10) #nrow(df2)=683, 10 plis/folds/K
observed <- c()
predicted <- c()

for (i in 1:folds$K) { #1:folds$K -> 1  2  3  4  5  6  7  8  9 10
  # Créer les ensembles d'entraînement et de test
  test_indices <- which(folds$which == i)
  train_indices <- setdiff(1:nrow(df2), test_indices)
  
  train_data <-  df2[train_indices, ] #615 observations
  test_data <-  df2[test_indices, ] #68 obs
  
  # Ajuster le modèle sur l'ensemble d'entraînement
  res_picto <- glmmTMB(annotator1_picto_all ~ translation + sentence_type + pos + V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = train_data, family = binomial)
  
  # Prédire les valeurs sur l'ensemble de test
  predictions <- predict(res_picto, newdata = test_data, type = "response") #n=68
  
  # Convertir les probabilités en étiquettes binaires (0 ou 1)
  predicted_labels <- ifelse(predictions > 0.5, 1, 0) #n=68
  
  # Stocker les étiquettes observées et prédites
  observed <- c(observed, test_data$annotator1_picto_all) #683
  predicted <- c(predicted, predicted_labels) #683
}

conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(observed))
print(conf_matrix)
prop.table(table(as.factor(predicted), as.factor(observed)))*100

##phrase

set.seed(1234) #pour la reproductibilité
folds <- cvFolds(nrow(df_sentence), K = 10) #nrow(df_sentence)
observed <- c()
predicted <- c()

for (i in 1:folds$K) {
  # Créer les ensembles d'entraînement et de test
  test_indices <- which(folds$which == i)
  train_indices <- setdiff(1:nrow(df_sentence), test_indices)
  
  train_data <-  df_sentence[train_indices, ]
  test_data <-  df_sentence[test_indices, ]
  
  # Ajuster le modèle sur l'ensemble d'entraînement
  res_sentence <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + level + (1|participant), data = train_data, family = binomial)
  
  # Prédire les valeurs sur l'ensemble de test
  predictions <- predict(res_sentence, newdata = test_data, type = "response")
  
  # Convertir les probabilités en étiquettes binaires (0 ou 1)
  predicted_labels <- ifelse(predictions > 0.5, 1, 0)
  
  # Stocker les étiquettes observées et prédites
  observed <- c(observed, test_data$annotator1_sentence)
  predicted <- c(predicted, predicted_labels)
}

conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(observed))
print(conf_matrix)
prop.table(table(as.factor(predicted), as.factor(observed)))*100
