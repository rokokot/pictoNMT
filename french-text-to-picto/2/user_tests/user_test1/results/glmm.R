library(readr) #read_tsv()
library(psych)
library(dplyr)
library(ggplot2)

library(glmmTMB) #mod <- glmmTMB()
library(lme4) #mod <- lme4::glmer()

library(DHARMa) #plot(DHARMa::simulateResiduals()) #https://cran.r-project.org/web/packages/DHARMa/vignettes/DHARMa.html
library(performance) #performance::check_collinearity() -> multicolinéarité : vif, 95 % intervalle confiance

library(car) #Anova
library(emmeans) #
library(effects) #plot(allEffects(res))
library(caret) #predict()

df <- read_tsv(file = "results_test1.tsv") #773 picto
df_sentence <- df %>% distinct(n_sentence, .keep_all = T) #180 sentences #nécessite psych et dplyr

cor.test(df_sentence$annotator1_sentence, df_sentence$annotator1_picto_all2)
#Pearson's product-moment correlation
#data:  df_sentence$annotator1_sentence and df_sentence$annotator1_picto_all2
#t = 11.822, df = 178, p-value < 2.2e-16
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# 0.5724566 0.7378799
#sample estimates:
#      cor 
#0.6631913

boxplot(df_sentence$annotator1_picto_all2~df_sentence$annotator1_sentence, xlab="A1-Phrase", ylab="A1-PictoParPhrase") #boxplot

cor.test(df_sentence$annotator1_sentence, df_sentence$V1)
boxplot(df_sentence$V1~df_sentence$annotator1_sentence, xlab="A1-Phrase", ylab="V1") #boxplot, nbre de picto par phrase (2-8)
boxplot(df_sentence$V1~df_sentence$annotator1_sentence, xlab="A1-Phrase", ylab="Nombre de pictogrammes par phrase")

#PICTO

#mod 1, 3-7, level + (1|participant)
r1 <- glmmTMB(annotator1_picto_all ~ participant, data = df, family = binomial)

r2 <- glmmTMB(annotator1_picto_all ~ level + (1|participant), data = df, family = binomial)
r3 <- glmmTMB(annotator1_picto_all ~ translation + level + (1|participant), data = df, family = binomial)
r4 <- glmmTMB(annotator1_picto_all ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = df, family = binomial)
r5 <- glmmTMB(annotator1_picto_all ~ sentence_type + level + (1|participant), data = df, family = binomial)
r6 <- glmmTMB(annotator1_picto_all ~ pos + level + (1|participant), data = df, family = binomial)

#mod 9-12, + (1|participant)
r1 <- glmmTMB(annotator1_picto_all ~ translation + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ sentence_type + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ pos + (1|participant), data = df, family = binomial)

plot(simulateResiduals(r1))
summary(r1)
exp(confint(r1))
car::Anova(r1)
performance::check_collinearity(r1)

#interaction entre variables fixes (hypothèse étudiée picto/phrase vs level caractéristique de l'apprenant)
r1 <- glmmTMB(annotator1_picto_all ~ translation * level + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ V2 * V3 * V4 * V5 * V6 * V7 * V8 * level + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ sentence_type * level + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ pos * level + (1|participant), data = df, family = binomial)

#mod 15-16, + (1|participant)
r1 <- glmmTMB(annotator1_picto_all ~ translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + level + (1|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + pos + level + (1|participant), data = df, family = binomial)

r1 <- glmmTMB(annotator1_sentence ~ V1 + level + (1|participant), data = df_sentence, family = binomial)

#mod 17, 19-22, niveau + (niveau|participant) : pente aléatoire avec intercept corrélé
r1 <- glmmTMB(annotator1_picto_all ~ level + (level|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ translation + level + (level|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (level|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ sentence_type + level + (level|participant), data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ pos + level + (level|participant), data = df, family = binomial)

summary(r1)

#mod16 best mod picto
r1 <- glmmTMB(annotator1_picto_all ~ translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + pos + level + (1|participant), data = df, family = binomial)
df2 <- subset(df, select = c(annotator1_picto_all, participant, source_sentence, picto_evaluated, translation, level, V2, V3, V4, V5, V6, V7, V8, sentence_type, pos))
df2_picto <- na.omit(df2) #683 picto
r1 <- glmmTMB(annotator1_picto_all ~ translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + pos + level + (1|participant), data = df2_picto, family = binomial)
predictions_picto <- predict(r1, type = "response") #nécessite glmmTMB
predictions_picto <- ifelse(predictions_picto>.5,1,0)
confusionMatrix(table(predictions_picto, df2_picto$annotator1_picto_all)) #nécessite caret
prop.table(table(predictions_picto, df2_picto$annotator1_picto_all))*100
#> cor.test(predictions_picto, df2_picto$annotator1_picto_all)
#Pearson's product-moment correlation

#data:  predictions_picto and df2_picto$annotator1_picto_all
#t = 10.274, df = 681, p-value < 2.2e-16
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# 0.2995560 0.4295577
#sample estimates:
#      cor 
#0.3663433 

#uniquement la variable de l'hypothèse étudiée, aic et bic plus élevés
r1 <- glmmTMB(annotator1_picto_all ~ level, data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ translation, data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ V2 + V3 + V4 + V5 + V6 + V7 + V8, data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ V2 * V3 * V4 * V5 * V6 * V7 * V8, data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ sentence_type, data = df, family = binomial)
r1 <- glmmTMB(annotator1_picto_all ~ pos, data = df, family = binomial)

#PHRASE

#mod 1-7, level + (1|participant)
r1 <- glmmTMB(annotator1_sentence ~ participant, data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ translation + level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ sentence_type + level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ pos + level + (1|participant), data = df_sentence, family = binomial)
#r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + pos + level + (1|participant), data = df_sentence, family = binomial) #Model convergence problem

#mod 8-12, + (1|participant)
r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ translation + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ sentence_type + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ pos + (1|participant), data = df_sentence, family = binomial)

#mod 13-16, + (1|participant)
r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + translation + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + translation + level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + level + (1|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ translation + V2 + V3 + V4 + V5 + V6 + V7 + V8 + sentence_type + pos + level + (1|participant), data = df_sentence, family = binomial)

#mod 17-22, niveau + (niveau|participant) : pente aléatoire avec intercept corrélé
r1 <- glmmTMB(annotator1_sentence ~ level + (level|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + (level|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ translation + (level|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + (level|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ sentence_type + (level|participant), data = df_sentence, family = binomial)
r1 <- glmmTMB(annotator1_sentence ~ pos + (level|participant), data = df_sentence, family = binomial)

summary(r1)

#mod2 best mod phrase
r1 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + level + (1|participant), data = df_sentence, family = binomial)
predictions_phrase <- predict(r1, type = "response") #nécessite glmmTMB
predictions_phrase <- ifelse(predictions_phrase>.5,1,0)
confusionMatrix(table(predictions_phrase, df_sentence$annotator1_sentence)) #nécessite caret
prop.table(table(predictions_phrase, df_sentence$annotator1_sentence))*100
#> cor.test(predictions_phrase, df_sentence$annotator1_sentence)
#Pearson's product-moment correlation

#data:  predictions_phrase and df_sentence$annotator1_sentence
#t = 14.085, df = 178, p-value < 2.2e-16
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# 0.6486311 0.7885446
#sample estimates:
#      cor 
#0.7260165

##Par modalité (one-hot encoding)

#PICTO

df0 <- subset(df, select = c(id, participant, annotator1_picto_all, translation, level, V2, V3, V4, V5, V6, V7, V8, sentence_type, pos))
dmy <- dummyVars(" ~ .", data = df0) #nécessite caret
df2 <- data.frame(predict(dmy, newdata = df0))

#effets aléatoires (P1 -> P9, pas besoin de vérifier)
m1 <- glmmTMB(annotator1_picto_all ~ participantP1, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ participantP2, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ participantP3, data = df2, family = binomial)

m1 <- glmmTMB(annotator1_picto_all ~ levelA1, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ levelA2, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ levelB1, data = df2, family = binomial)

m1 <- glmmTMB(annotator1_picto_all ~ translationtexttopicto, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ translationpictodr, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V2, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V3, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V4, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V5, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V6, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V7, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ V8, data = df2, family = binomial)

m1 <- glmmTMB(annotator1_picto_all ~ sentence_typeinstruction, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ sentence_typequestion, data = df2, family = binomial)

m1 <- glmmTMB(annotator1_picto_all ~ posadj, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ posadv, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ posinterj, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ posn, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ pospro, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ posverb, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_picto_all ~ postp, data = df2, family = binomial)

df_participant <- subset(df, select = c(id, participant))
df_picto <- merge(df2, df_participant, by="id")
m1 <- glmmTMB(annotator1_picto_all ~ levelA1 + levelB1 + translationtexttopicto + translationpictodr + V2 + sentence_typeinstruction + sentence_typequestion + posn + posverb + (1|participant), data = df_picto, family = binomial)
df2_picto <- na.omit(df_picto) #683 picto
m1 <- glmmTMB(annotator1_picto_all ~ levelA1 + levelB1 + translationtexttopicto + translationpictodr + V2 + sentence_typeinstruction + sentence_typequestion + posn + posverb + (1|participant), data = df2_picto, family = binomial)
predictions_picto <- predict(m1, type = "response") #nécessite glmmTMB
predictions_picto <- ifelse(predictions_picto>.5,1,0)
confusionMatrix(table(predictions_picto, df2_picto$annotator1_picto_all)) #accuracy : 0.6706
prop.table(table(predictions_picto, df2_picto$annotator1_picto_all))*100

df_participant <- subset(df, select = c(id, participant, translation, sentence_type))
df_picto <- merge(df2, df_participant, by="id")
m1 <- glmmTMB(annotator1_picto_all ~ levelA1 + levelB1 + V2 + posn + posverb + (1|participant), data = df_picto, family = binomial)
df2_picto <- na.omit(df_picto) #683 picto
m1 <- glmmTMB(annotator1_picto_all ~ levelA1 + levelB1 + translation + V2 + sentence_type + posn + posverb + (1|participant), data = df2_picto, family = binomial)
predictions_picto <- predict(m1, type = "response") #nécessite glmmTMB
predictions_picto <- ifelse(predictions_picto>.5,1,0)
confusionMatrix(table(predictions_picto, df2_picto$annotator1_picto_all)) #accuracy : 0.6706
prop.table(table(predictions_picto, df2_picto$annotator1_picto_all))*100

plot(simulateResiduals(m1))
summary(m1)
exp(confint(m1))
car::Anova(m1)
performance::check_collinearity(m1)

#PHRASE

df02 <- subset(df_sentence, select = c(id, participant, annotator1_sentence, translation, level, V2, V3, V4, V5, V6, V7, V8, sentence_type, pos))
dmy <- dummyVars(" ~ .", data = df02) #nécessite caret
df2 <- data.frame(predict(dmy, newdata = df02))

m1 <- glmmTMB(annotator1_sentence ~ levelA1, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ levelA2, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ levelB1, data = df2, family = binomial)

m1 <- glmmTMB(annotator1_sentence ~ translationtexttopicto, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ translationpictodr, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ V2, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ V3, data = df2, family = binomial) #**
m1 <- glmmTMB(annotator1_sentence ~ V4, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ V5, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ V6, data = df2, family = binomial)
m1 <- glmmTMB(annotator1_sentence ~ V7, data = df2, family = binomial) #*
m1 <- glmmTMB(annotator1_sentence ~ V8, data = df2, family = binomial)

m1 <- glmmTMB(annotator1_sentence ~ sentence_typeinstruction, data = df2, family = binomial) #***
m1 <- glmmTMB(annotator1_sentence ~ sentence_typequestion, data = df2, family = binomial) #***

df_participant <- subset(df, select = c(id, participant))
df_picto <- merge(df2, df_participant, by="id")

m1 <- glmmTMB(annotator1_sentence ~ V3 + V7 + sentence_typeinstruction + sentence_typequestion + (1|participant), data = df_picto, family = binomial) #***
df2_picto <- na.omit(df_picto) #683 picto
m1 <- glmmTMB(annotator1_sentence ~ V3 + V7 + sentence_typeinstruction + sentence_typequestion + (1|participant), data = df2_picto, family = binomial) #***
predictions_picto <- predict(m1, type = "response") #nécessite glmmTMB
predictions_picto <- ifelse(predictions_picto>.5,1,0)
confusionMatrix(table(predictions_picto, df2_picto$annotator1_sentence)) #accuracy : 0.7889
prop.table(table(predictions_picto, df2_picto$annotator1_sentence))*100

plot(simulateResiduals(m1))
summary(m1)
exp(confint(m1))
car::Anova(m1)
performance::check_collinearity(m1)