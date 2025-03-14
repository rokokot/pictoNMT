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

#données : picto / phrases
df <- read_tsv(file = "results_test1.tsv") #773 picto
df_sentence <- df %>% distinct(n_sentence, .keep_all = T) #180 sentences #nécessite psych et dplyr

#dataframe : 90 phrases de pictodr, toutes les colonnes du df_sentence
df_pictodr <- subset(df_sentence, translation=="pictodr")

#test prédiction - picto
#création d'un df avec les colonnes utilisées dans le mixed model auquel on enlève les lignes qui ont une colonne avec au moins un na
df2 <- subset(df, select = c(annotator1_picto_all, participant, source_sentence, picto_evaluated, translation, level, V2, V3, V4, V5, V6, V7, V8, sentence_type, pos))
#df2 <- subset(df, select = c(annotator1_picto_all, participant, source_sentence, picto_evaluated, translation, level, V2, V3, V4, V5, V6, V7, V8, participant))
df2 <- na.omit(df2) #683 picto
res2 <- glmmTMB(annotator1_picto_all ~ translation + sentence_type + pos + V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = df2, family = binomial)
#res2 <- glmmTMB(annotator1_picto_all ~ translation + level + V2 + V3 + V4 + V5 + V6 + V7 + V8 + (1|participant), data = df2, family = binomial)
predictions_picto <- predict(res2, type = "response") #nécessite glmmTMB
predictions_picto <- ifelse(predictions_picto>.5,1,0)
confusionMatrix(table(predictions_picto, df2$annotator1_picto_all)) #nécessite caret
prop.table(table(predictions_picto, df2$annotator1_picto_all))*100
plot(predictions)

#test prédiction - phrase
df2_sentence <- subset(df_sentence, select = c(annotator1_sentence, participant, source_sentence, picto_evaluated, translation, level, V2, V3, V4, V5, V6, V7, V8, participant, sentence_type, pos))
#df2_sentence <- subset(df_sentence, select = c(annotator1_sentence, translation, level, V2, V3, V4, V5, V6, V7, V8, participant))
df2_sentence <- na.omit(df2_sentence) #683 picto
res2_sentence <- glmmTMB(annotator1_sentence ~ translation + sentence_type + pos + V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = df2_sentence, family = binomial)
#res2_sentence <- glmmTMB(annotator1_sentence ~ translation + level + V2 + V3 + V4 + V5 + V6 + V7 + V8 + (1|participant), data = df2_sentence, family = binomial)
predictions_sentence <- predict(res2_sentence, type = "response") #nécessite glmmTMB
predictions_sentence <- ifelse(predictions_sentence>.5,1,0)
confusionMatrix(table(predictions_sentence, df2_sentence$annotator1_sentence)) #nécessite caret
prop.table(table(predictions_sentence, df2_sentence$annotator1_sentence))*100
plot(predictions_sentence)

#One-hot encoding : créer un new dataframe variables catégorielles -> binaires (0-1)
dmy <- dummyVars(" ~ .", data = df) #nécessite caret
df2 <- data.frame(predict(dmy, newdata = df))

#picto vs phrase (11bb)
#test d'un modèle mixte avec le pourcentage de picto correct par phrase (annotator1_picto_all2)
df$translation <- factor(df$translation)
df$translation <- relevel(df$translation, "texttopicto") #texttopicto
res <- glmmTMB(annotator1_picto_all ~ annotator1_picto_all2, data = df_sentence, family = binomial)
res <- glmmTMB(annotator1_picto_all ~ annotator1_picto_all2 + level + (1|participant), data = df_sentence, family = binomial)
#res_i <- glmmTMB(annotator1_picto_all ~ annotator1_picto_all2*translation + level + (1|participant), data = df_sentence, family = binomial)
res <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + translation + pos + level + (1|participant), data = df_sentence, family = binomial)
#res2 <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2 + level, data = df_sentence, family = binomial)
#res_i <- glmmTMB(annotator1_sentence ~ annotator1_picto_all2*translation + level + (1|participant), data = df_sentence, family = binomial)
summary(res)
#summary(res2)
#summary(res_i)
exp(confint(res))
car::Anova(res)

#factor (variables indep catégorielles)
df$participant <- factor(df$participant)
df$level <- factor(df$level)
df$translation <- factor(df$translation)
#df$V2 <- factor(df$V2)
df$sentence_type <- factor(df$sentence_type)
df$pos <- factor(df$pos)

#changement modalité de référence
df$translation <- factor(df$translation)
df$translation <- relevel(df$translation, "texttopicto") #texttopicto
df$pos <- factor(df$pos)
df$pos <- relevel(df$pos, "tp") #temps

#Picto (a)

#picto vs phrase (11a/11b)
#res <- glmmTMB(annotator1_picto_all ~ annotator1_sentence, data = df, family = binomial)
#res <- glmmTMB(annotator1_picto_all ~ annotator1_sentence + (1|participant), data = df, family = binomial)
res <- glmmTMB(annotator1_picto_all ~ annotator1_sentence + level + (1|participant), data = df, family = binomial)
res <- glmmTMB(annotator1_sentence ~ annotator1_picto_all + level + (1|participant), data = df, family = binomial)
plot(simulateResiduals(res))
summary(res)
confint(res)
car::Anova(res)

#participant (12a)
#res_picto_participant <- glmmTMB(annotator1_picto_all ~ (1|participant), data = df, family = binomial)
res_picto_participant <- glmmTMB(annotator1_picto_all ~ participant, data = df, family = binomial)
#level ici sert à rien
summary(res_picto_participant)
confint(res_picto_participant)
car::Anova(res_picto_participant)
exp(confint(res_picto_participant))
plot(simulateResiduals(res_picto_participant))

#level (2a)
res_picto_level <- glmmTMB(annotator1_picto_all ~ level + (1|participant), data = df, family = binomial(link="logit"))
plot(simulateResiduals(res_picto_level))
summary(res_picto_level)
confint(res_picto_level)
car::Anova(res_picto_level)
exp(confint(res_picto_level))

#level + translation (3a)
res_picto_translation_level <- glmmTMB(annotator1_picto_all ~ translation + level + (1|participant), data = df, family = binomial)
res_picto_translation_level_i <- glmmTMB(annotator1_picto_all ~ translation*level + (1|participant), data = df, family = binomial)
res_picto_translation <- glmmTMB(annotator1_picto_all ~ translation + (1|participant), data = df, family = binomial)
summary(res_picto_translation_level)
summary(res_picto_translation_level_i)
plot(simulateResiduals(res_picto_translation_level))
confint(res_picto_translation_level)
confint(res_picto_translation_level_i)
car::Anova(res_picto_translation_level)
car::Anova(res_picto_translation_level_i)
exp(confint(res_picto_translation_level))

res_picto_translation <- glmmTMB(annotator1_picto_all ~ translation + (1|participant), data = df, family = binomial)
summary(res_picto_translation)
confint(res_picto_translation)
car::Anova(res_picto_translation)
car::qqPlot(residuals(res_picto_translation_level))

#level + var (3ab)
res_picto_var_level <- glmmTMB(annotator1_picto_all ~ v1 + v2 + v3 + v4 + v5 + level + (1|participant), data = df, family = binomial)
res_picto_var_level <- glmmTMB(annotator1_picto_all ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = df, family = binomial)
res_picto_var_translation_level <- glmmTMB(annotator1_picto_all ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + translation + level + (1|participant), data = df, family = binomial)
summary(res_picto_var_level)
summary(res_picto_var_translation_level)
car::Anova(res_picto_var_level)

#level + sentence_type (4a)
res_picto_sentencetype <- glmmTMB(annotator1_picto_all ~ sentence_type + level + (1|participant), data = df, family = binomial)
summary(res_picto_sentencetype)
confint(res_picto_sentencetype)
car::Anova(res_picto_sentencetype)
exp(confint(res_picto_sentencetype))

#level + pos (5a)
df$pos <- factor(df$pos)
df$pos <- relevel(df$pos, "adj")
df$pos <- relevel(df$pos, "adv")
df$pos <- relevel(df$pos, "interj")
df$pos <- relevel(df$pos, "n")
df$pos <- relevel(df$pos, "pro")
df$pos <- relevel(df$pos, "verb")
df$pos <- relevel(df$pos, "tp")
res_picto_pos <- glmmTMB(annotator1_picto_all ~ pos + level + (1|participant), data = df, family = binomial)
summary(res_picto_pos)
confint(res_picto_pos)
exp(confint(res_picto_pos))
car::Anova(res_picto_pos)

res <- glmmTMB(annotator1_picto_all ~ translation + pos + level + (1|participant), data = df, family = binomial)
summary(res)

#level + translation + sentence_type (a)
res_picto_translation_sentencetype_level <- glmmTMB(annotator1_picto_all ~ translation + sentence_type + level + (1|participant), data = df, family = binomial)
summary(res_picto_translation_sentencetype_level)
confint(res_picto_translation_sentencetype_level)
car::Anova(res_picto_translation_sentencetype_level)

#level + sentence_type + pos (a)
res_picto_translation_sentencetype_pos_level <- glmmTMB(annotator1_picto_all ~ sentence_type + pos + level + (1|participant), data = df, family = binomial)

#level + translation + sentence_type + pos
res_picto_translation_sentencetype_pos_level <- glmmTMB(annotator1_picto_all ~ translation + sentence_type + pos + level + (1|participant), data = df, family = binomial)
summary(res_picto_translation_sentencetype_pos_level)
confint(res_picto_translation_sentencetype_pos_level)
car::Anova(res_picto_translation_sentencetype_pos_level)
plot(simulateResiduals(res_picto_translation_sentencetype_pos_level))

#-----

#Phrase (b)

#picto vs phrase (11b)
res <- glmmTMB(annotator1_sentence ~ annotator1_picto_all, data = df, family = binomial)
res <- glmmTMB(annotator1_sentence ~ annotator1_picto_all + (1|participant), data = df, family = binomial)
res <- glmmTMB(annotator1_sentence ~ annotator1_picto_all + level + (1|participant), data = df, family = binomial)
summary(res)
confint(res)
car::Anova(res)

#participant (12b)
res_sentence_participant <- glmmTMB(annotator1_sentence ~ participant, data = df_sentence, family = binomial)
summary(res_sentence_participant)
confint(res_sentence_participant)

#level (2b)
res_sentence_level <- glmmTMB(annotator1_sentence ~ level + (1|participant), data = df_sentence, family = binomial)
summary(res_sentence_level)
confint(res_sentence_level)
car::Anova(res_sentence_level)
exp(confint(res_sentence_level))

#level + translation (3b)
df$translation <- relevel(df$translation, "texttopicto")
res_sentence_translation <- glmmTMB(annotator1_sentence ~ translation + level + (1|participant), data = df_sentence, family = binomial)
summary(res_sentence_translation)
confint(res_sentence_translation)
car::Anova(res_sentence_translation)
exp(confint(res_sentence_level))

#level + var (3bb)
res_sentence_var_level <- glmmTMB(annotator1_sentence ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + level + (1|participant), data = df_sentence, family = binomial)
res_sentence_var_level <- glmmTMB(annotator1_sentence ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + level + (1|participant), data = df_sentence, family = binomial)
summary(res_sentence_var_level)
confint(res_sentence_var_level)
car::Anova(res_sentence_var_level)
exp(confint(res_sentence_var_level))

#level + sentence_type (4b)
res_picto_sentencetype <- glmmTMB(annotator1_sentence ~ sentence_type + level + (1|participant), data = df_sentence, family = binomial)
summary(res_sentence_sentencetype)
confint(res_sentence_sentencetype)
car::Anova(res_sentence_sentencetype)
exp(confint(res_sentence_sentencetype))

#level + pos (5b)
res_sentence_pos <- glmmTMB(annotator1_sentence ~ pos + level + (1|participant), data = df_sentence, family = binomial)
summary(res_sentence_pos)
confint(res_sentence_pos)
car::Anova(res_sentence_pos)
exp(confint(res_sentence_pos))

res_picto_translation <- glmmTMB(annotator1_picto_all ~ translation + sentence_type + pos + level + (1|participant), data = df, family = binomial)

#picto - translation/level
res_picto_translation <- glmmTMB(annotator1_picto_all ~ translation + level + V2 + V3 + V4 + V5 + V6 + V7 + V8 + (1|participant), data = df, family = binomial)
summary(res_picto_translation)
as.data.frame(VarCorr(res_picto_translation)$cond$participant)
confint(res_picto_translation)
exp(confint(res_picto_translation)[2,])
car::Anova(res_picto_translation)

res_sentence_translation <- glmmTMB(annotator1_sentence ~ translation + level + V2 + V3 + V4 + V5 + V6 + V7 + V8 + (1|participant), data = df_sentence, family = binomial)
summary(res_sentence_translation)

res_picto_translation <- glmmTMB(annotator1_picto_all ~ sentence_type + level + (1|participant), data = df, family = binomial)