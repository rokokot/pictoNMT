library(readr)
library(psych)
library(dplyr)
library(forcats) #fct_recode()
library(irr)

df <- read_tsv(file = "results_test1.tsv") #773 picto
df_sentence <- df %>% distinct(n_sentence, .keep_all = T) #180 sentences #nécessite psych et dplyr

names(df)
summary(df)
describe(df)

#obtenir les lignes des colonnes participant et response_participant où le picto évalué (colonne) est "passé"
subset(df, picto_evaluated=="passé", c(participant, response_participant))

#effectif de chaque pos
table(df$pos) #adj    adv interj      n    pro     tp   verb -> 35     80     18    130    183     31    206
subset(df, pos=="interj", c(participant, picto_evaluated))
print(n=683, subset(df, pos=="adj", c(participant, picto_evaluated, translation, annotator1_picto_all)))

subset(df, picto_evaluated=="causer", c(participant, picto_evaluated, annotator1_sentence))

## H1 : compréhension pictogrammes et phrases par participant (Px), annotateur (A1/A2), picto/phrase

#t-test : significativité entre A1/A2-picto (#180) et A1/A2-phrase (#180) -> non
a1_picto <- c(40,60,45,50,30,25,30,5,25) #cf. infra
a2_picto <- c(40,25,35,40,5,15,30,0,35)
a_picto <- matrix(data = c(a1_picto, a2_picto), nrow = 2, byrow = TRUE)
a_picto_m <- colMeans(a_picto)
a1_phrase <- c(35,45,35,15,20,25,25,0,30)
a2_phrase <- c(40,55,35,20,5,25,25,0,20)
a_phrase <- matrix(data = c(a1_phrase, a2_phrase), nrow = 2, byrow = TRUE)
a_phrase_m <- colMeans(a_phrase)
t.test(a_picto_m, a_phrase_m, paired = TRUE) #t = 1.3571, df = 8, p-value = 0.2118
#t-test : significativité entre A1-picto (#683)/A2-picto (#180) et A1/A2-phrase (#180) ? -> oui
a1_picto_all_m <- c(39.75904,54.21687,60.24096,51.80723,32.53012,40.29851,49.25373,10.44776,47.76119)
t.test(a1_picto_all_m, a_phrase_m, paired = TRUE) #t = 5.0326, df = 8, p-value = 0.001011

#test de proportion : significativité entre A1/A2-picto (#180) et A1/A2-phrase (#180) ? -> non
a1_picto <- table(df$annotator1_picto)
a2_picto <- table(df$annotator2_picto)
a1_sentence <- table(df_sentence$annotator1_sentence)
a2_sentence <- table(df_sentence$annotator2_sentence)
prop.test(x = c(a1_picto["1"]+a2_picto["1"], a1_sentence["1"]+a2_sentence["1"]), n = c(360,360))
t.test(a1_picto, a2_picto, paired = FALSE)

#significativité entre A1-picto (#683)/A2-picto (#180) et A1/A2-phrase (#180) ? -> oui p-value = 2.388e-06 **
a1_picto <- table(df$annotator1_picto_all)
a2_picto <- table(df$annotator2_picto)
a1_sentence <- table(df_sentence$annotator1_sentence)
a2_sentence <- table(df_sentence$annotator2_sentence)
prop.test(x = c(a1_picto["1"]+a2_picto["1"], a1_sentence["1"]+a2_sentence["1"]), n = c(863,360))

#proportion de réponses correctes pour tous les pictos (#683) avec A1
p1 <- prop.table(table(subset(df, participant=="P1", annotator1_picto_all)))*100
p2 <- prop.table(table(subset(df, participant=="P2", annotator1_picto_all)))*100
p3 <- prop.table(table(subset(df, participant=="P3", annotator1_picto_all)))*100
p4 <- prop.table(table(subset(df, participant=="P4", annotator1_picto_all)))*100
p5 <- prop.table(table(subset(df, participant=="P5", annotator1_picto_all)))*100
p6 <- prop.table(table(subset(df, participant=="P6", annotator1_picto_all)))*100
p7 <- prop.table(table(subset(df, participant=="P7", annotator1_picto_all)))*100
p8 <- prop.table(table(subset(df, participant=="P8", annotator1_picto_all)))*100
p9 <- prop.table(table(subset(df, participant=="P9", annotator1_picto_all)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#39.75904 54.21687 60.24096 51.80723 32.53012 40.29851 49.25373 10.44776 47.76119
mean(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],p8[2],p9[2])) #42.92393
sd(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],p8[2],p9[2])) #14.78779

#proportion de réponses correctes pour un picto par phrase (#180) avec A1
p1 <- prop.table(table(subset(df, participant=="P1", annotator1_picto)))*100
p2 <- prop.table(table(subset(df, participant=="P2", annotator1_picto)))*100
p3 <- prop.table(table(subset(df, participant=="P3", annotator1_picto)))*100
p4 <- prop.table(table(subset(df, participant=="P4", annotator1_picto)))*100
p5 <- prop.table(table(subset(df, participant=="P5", annotator1_picto)))*100
p6 <- prop.table(table(subset(df, participant=="P6", annotator1_picto)))*100
p7 <- prop.table(table(subset(df, participant=="P7", annotator1_picto)))*100
p8 <- prop.table(table(subset(df, participant=="P8", annotator1_picto)))*100
p9 <- prop.table(table(subset(df, participant=="P9", annotator1_picto)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#40 60 45 50 30 25 30  5 25
mean(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],p8[2],p9[2])) #34.44444
sd(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],p8[2],p9[2])) #16.28735

#proportion de réponses correctes pour un picto par phrase (#180) avec A2 (PROP2)
p1 <- prop.table(table(subset(df, participant=="P1", annotator2_picto)))*100
p2 <- prop.table(table(subset(df, participant=="P2", annotator2_picto)))*100
p3 <- prop.table(table(subset(df, participant=="P3", annotator2_picto)))*100
p4 <- prop.table(table(subset(df, participant=="P4", annotator2_picto)))*100
p5 <- prop.table(table(subset(df, participant=="P5", annotator2_picto)))*100
p6 <- prop.table(table(subset(df, participant=="P6", annotator2_picto)))*100
p7 <- prop.table(table(subset(df, participant=="P7", annotator2_picto)))*100
p8 <- prop.table(table(subset(df, participant=="P8", annotator2_picto)))*100
p9 <- prop.table(table(subset(df, participant=="P9", annotator2_picto)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#40   25   35   40    5   15   30   NA   35
mean(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],0,p9[2])) #25
sd(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],0,p9[2])) #15

#si fusion 0 et 'na'
df$annotator2_picto <- factor(df$annotator2_picto)
levels(df$annotator2_picto) #"0"  "1"  "na"
levels(df$annotator2_picto) <- factor(c("0", "1", "NA"))

df$annotator2_picto <- fct_recode(df$annotator2_picto, NULL = "NA")

#proportion de réponses correctes pour un picto par phrase (#180) avec A2 (PROP3)
p1 <- prop.table(table(subset(df, participant=="P1", annotator2_picto)))*100
p2 <- prop.table(table(subset(df, participant=="P2", annotator2_picto)))*100
p3 <- prop.table(table(subset(df, participant=="P3", annotator2_picto)))*100
p4 <- prop.table(table(subset(df, participant=="P4", annotator2_picto)))*100
p5 <- prop.table(table(subset(df, participant=="P5", annotator2_picto)))*100
p6 <- prop.table(table(subset(df, participant=="P6", annotator2_picto)))*100
p7 <- prop.table(table(subset(df, participant=="P7", annotator2_picto)))*100
p8 <- prop.table(table(subset(df, participant=="P8", annotator2_picto)))*100
p9 <- prop.table(table(subset(df, participant=="P9", annotator2_picto)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#53.33333 31.25000 36.84211 47.05882  5.00000 20.00000 31.57895  0.00000 36.84211
mean(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],0,p9[2])) #29.10059

#proportion de réponses correctes pour toutes les phrases (#180) avec A1
p1 <- prop.table(table(subset(df_sentence, participant=="P1", annotator1_sentence)))*100
#p1 <- prop.table(table(subset(df_sentence, participant=="P1" & translation=="pictodr" & sentence_type=="question", annotator1_sentence)))*100
p2 <- prop.table(table(subset(df_sentence, participant=="P2", annotator1_sentence)))*100
p3 <- prop.table(table(subset(df_sentence, participant=="P3", annotator1_sentence)))*100
p4 <- prop.table(table(subset(df_sentence, participant=="P4", annotator1_sentence)))*100
p5 <- prop.table(table(subset(df_sentence, participant=="P5", annotator1_sentence)))*100
p6 <- prop.table(table(subset(df_sentence, participant=="P6", annotator1_sentence)))*100
p7 <- prop.table(table(subset(df_sentence, participant=="P7", annotator1_sentence)))*100
p8 <- prop.table(table(subset(df_sentence, participant=="P8", annotator1_sentence)))*100
p9 <- prop.table(table(subset(df_sentence, participant=="P9", annotator1_sentence)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#35   45   35   15   20   25   25   NA   30
mean(c(p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2],0,p9[2])) #25.55556

#proportion de réponses correctes pour les phrases (#180) avec A2 (PROP2)
p1 <- prop.table(table(subset(df_sentence, participant=="P1", annotator2_sentence)))*100
p2 <- prop.table(table(subset(df_sentence, participant=="P2", annotator2_sentence)))*100
p3 <- prop.table(table(subset(df_sentence, participant=="P3", annotator2_sentence)))*100
p4 <- prop.table(table(subset(df_sentence, participant=="P4", annotator2_sentence)))*100
p5 <- prop.table(table(subset(df_sentence, participant=="P5", annotator2_sentence)))*100
p6 <- prop.table(table(subset(df_sentence, participant=="P6", annotator2_sentence)))*100
p7 <- prop.table(table(subset(df_sentence, participant=="P7", annotator2_sentence)))*100
p8 <- prop.table(table(subset(df_sentence, participant=="P8", annotator2_sentence)))*100
p9 <- prop.table(table(subset(df_sentence, participant=="P9", annotator2_sentence)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#40   55   35   20    5   25   25   NA   20
mean(c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7[2],0,p9["1"])) #25
mean(c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7[2],p8["1"],p9["1"]), na.rm = TRUE) #28,125

cohen.kappa(x=cbind(df_sentence$annotator1_sentence,df_sentence$annotator2_sentence)) #0,57

#proportion de réponses correctes pour les phrases avec A2 (PROP3)
df_sentence$annotator2_sentence <- factor(df_sentence$annotator2_sentence)
levels(df_sentence$annotator2_sentence) <- factor(c("0", "1", "NA"))
df_sentence$annotator2_sentence <- fct_recode(df_sentence$annotator2_sentence, NULL = "NA")
p1 <- prop.table(table(subset(df_sentence, participant=="P1", annotator2_sentence)))*100
p2 <- prop.table(table(subset(df_sentence, participant=="P2", annotator2_sentence)))*100
p3 <- prop.table(table(subset(df_sentence, participant=="P3", annotator2_sentence)))*100
p4 <- prop.table(table(subset(df_sentence, participant=="P4", annotator2_sentence)))*100
p5 <- prop.table(table(subset(df_sentence, participant=="P5", annotator2_sentence)))*100
p6 <- prop.table(table(subset(df_sentence, participant=="P6", annotator2_sentence)))*100
p7 <- prop.table(table(subset(df_sentence, participant=="P7", annotator2_sentence)))*100
p8 <- prop.table(table(subset(df_sentence, participant=="P8", annotator2_sentence)))*100
p9 <- prop.table(table(subset(df_sentence, participant=="P9", annotator2_sentence)))*100
c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7["1"],p8["1"],p9["1"])
#47.058824 55.000000 38.888889 23.529412  5.263158 33.333333 27.777778  0.000000 21.052632
mean(c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7[2],0,p9["1"])) #27.98934
mean(c(p1["1"],p2["1"],p3["1"],p4["1"],p5["1"],p6["1"],p7[2],p8["1"],p9["1"]), na.rm = TRUE) #27.98934

## H2 : translation/system texttopicto/pictodr

#picto_all (A1)
texttopicto_picto_effectif <- table(subset(df, translation=="texttopicto", annotator1_picto_all))
pictodr_picto_effectif <- table(subset(df, translation=="pictodr", annotator1_picto_all))
texttopicto_picto <- prop.table(texttopicto_picto_effectif)*100
pictodr_picto <- prop.table(pictodr_picto_effectif)*100
c(texttopicto_picto["1"],pictodr_picto["1"]) #39.36430 49.63504
mean(c(texttopicto_picto["1"],pictodr_picto["1"])) #44.49967
prop.test(x = c(texttopicto_picto_effectif["1"], pictodr_picto_effectif["1"]), n = c((texttopicto_picto_effectif["1"]+texttopicto_picto_effectif["0"]), (pictodr_picto_effectif["1"]+pictodr_picto_effectif["0"]))) #p-value = 0.01002

#sentence (A1)
texttopicto_sentence_effectif <- table(subset(df_sentence, translation=="texttopicto", annotator1_sentence))
pictodr_sentence_effectif <- table(subset(df_sentence, translation=="pictodr", annotator1_sentence))
texttopicto_sentence <- prop.table(texttopicto_sentence_effectif)*100
pictodr_sentence <- prop.table(pictodr_sentence_effectif)*100
c(texttopicto_sentence["1"],pictodr_sentence["1"]) #20.00000 31.11111
mean(c(texttopicto_sentence["1"],pictodr_sentence["1"])) #25.55556
prop.test(x = c(texttopicto_sentence_effectif["1"], pictodr_sentence_effectif["1"]), n = c((texttopicto_sentence_effectif["1"]+texttopicto_sentence_effectif["0"]), (pictodr_sentence_effectif["1"]+pictodr_sentence_effectif["0"]))) #p-value = 0.1241

## H3 : level a1, a2, b1

#picto_all (A1)
df$level <- factor(df$level)
a1_picto_effectif <- table(subset(df, level=="A1", annotator1_picto_all))
a2_picto_effectif <- table(subset(df, level=="A2", annotator1_picto_all))
b1_picto_effectif <- table(subset(df, level=="B1", annotator1_picto_all))
a1_picto <- prop.table(a1_picto_effectif)*100
a2_picto <- prop.table(a2_picto_effectif)*100
b1_picto <- prop.table(b1_picto_effectif)*100
c(a1_picto["1"],a2_picto["1"],b1_picto["1"]) #33.33333 39.48498 55.42169
mean(c(a1_picto["1"],a2_picto["1"],b1_picto["1"])) #42.74667
#a1 >< a2 >< b1
prop.test(x = c(a1_picto_effectif["1"], a2_picto_effectif["1"], b1_picto_effectif["1"]), n = c((a1_picto_effectif["1"]+a1_picto_effectif["0"]), (a2_picto_effectif["1"]+a2_picto_effectif["0"]), (b1_picto_effectif["1"]+b1_picto_effectif["0"]))) #p-value = 5.075e-06
#a1 >< a2
prop.test(x = c(a1_picto_effectif["1"], a2_picto_effectif["1"]), n = c((a1_picto_effectif["1"]+a1_picto_effectif["0"]), (a2_picto_effectif["1"]+a2_picto_effectif["0"]))) #p-value = 0.22
#a2 >< b1
prop.test(x = c(a2_picto_effectif["1"], b1_picto_effectif["1"]), n = c((a2_picto_effectif["1"]+a2_picto_effectif["0"]), (b1_picto_effectif["1"]+b1_picto_effectif["0"]))) #p-value = 0.0006513

#sentence (A1)
a1_sentence_effectif <- table(subset(df_sentence, level=="A1", annotator1_sentence))
a2_sentence_effectif <- table(subset(df_sentence, level=="A2", annotator1_sentence))
b1_sentence_effectif <- table(subset(df_sentence, level=="B1", annotator1_sentence))
a1_sentence <- prop.table(a1_sentence_effectif)*100
a2_sentence <- prop.table(a2_sentence_effectif)*100
b1_sentence <- prop.table(b1_sentence_effectif)*100
c(a1_sentence["1"],a2_sentence["1"],b1_sentence["1"]) #16.66667 28.33333 31.66667
mean(c(a1_sentence["1"],a2_sentence["1"],b1_sentence["1"])) #25.55556
#a1 >< a2 >< b1
prop.test(x = c(a1_sentence_effectif["1"], a2_sentence_effectif["1"], b1_sentence_effectif["1"]), n = c((a1_sentence_effectif["1"]+a1_sentence_effectif["0"]), (a2_sentence_effectif["1"]+a2_sentence_effectif["0"]), (b1_sentence_effectif["1"]+b1_sentence_effectif["0"]))) #p-value = 0.1413
#a1 >< a2
prop.test(x = c(a1_sentence_effectif["1"], a2_sentence_effectif["1"]), n = c((a1_sentence_effectif["1"]+a1_sentence_effectif["0"]), (a2_sentence_effectif["1"]+a2_sentence_effectif["0"]))) #p-value = 0.1896
#a2 >< b1
prop.test(x = c(a2_sentence_effectif["1"], b1_sentence_effectif["1"]), n = c((a2_sentence_effectif["1"]+a2_sentence_effectif["0"]), (b1_sentence_effectif["1"]+b1_sentence_effectif["0"]))) #p-value = 0.8421

## H4 : type de phrases instruction, question

#picto_all (A1)
instruction_picto_effectif <- table(subset(df, sentence_type=="instruction", annotator1_picto_all))
question_picto_effectif <- table(subset(df, sentence_type=="question", annotator1_picto_all))
instruction_picto <- prop.table(instruction_picto_effectif)*100
question_picto <- prop.table(question_picto_effectif)*100
c(instruction_picto["1"],question_picto["1"]) #52.49267 34.50292
mean(c(instruction_picto["1"],question_picto["1"])) #43.4978
prop.test(x = c(instruction_picto_effectif["1"], question_picto_effectif["1"]), n = c((instruction_picto_effectif["1"]+instruction_picto_effectif["0"]), (question_picto_effectif["1"]+question_picto_effectif["0"]))) #p-value = 3.09e-06

#sentence (A1)
instruction_sentence_effectif <- table(subset(df_sentence, sentence_type=="instruction", annotator1_sentence))
question_sentence_effectif <- table(subset(df_sentence, sentence_type=="question", annotator1_sentence))
instruction_sentence <- prop.table(instruction_sentence_effectif)*100
question_sentence <- prop.table(question_sentence_effectif)*100
c(instruction_sentence["1"],question_sentence["1"]) #36.66667 14.44444 
mean(c(instruction_sentence["1"],question_sentence["1"])) #25.55556
prop.test(x = c(instruction_sentence_effectif["1"], question_sentence_effectif["1"]), n = c((instruction_sentence_effectif["1"]+instruction_sentence_effectif["0"]), (question_sentence_effectif["1"]+question_sentence_effectif["0"]))) #p-value = 0.001167

## H5 : POS adjectifs, adverbes, interjections, pronoms, noms, temps, verbes

#picto_all (A1)
df$pos <- factor(df$pos)
#& translation=="texttopicto" | & translation=="pictodr"
adj_picto_effectif <- table(subset(df, pos=="adj" & translation=="pictodr", annotator1_picto_all))
adv_picto_effectif <- table(subset(df, pos=="adv" & translation=="pictodr", annotator1_picto_all))
interj_picto_effectif <- table(subset(df, pos=="interj" & translation=="pictodr", annotator1_picto_all))
pro_picto_effectif <- table(subset(df, pos=="pro" & translation=="pictodr", annotator1_picto_all))
n_picto_effectif <- table(subset(df, pos=="n" & translation=="pictodr", annotator1_picto_all))
tp_picto_effectif <- table(subset(df, pos=="tp" & translation=="pictodr", annotator1_picto_all))
v_picto_effectif <- table(subset(df, pos=="verb" & translation=="pictodr", annotator1_picto_all))
adj_picto <- prop.table(adj_picto_effectif)*100
adv_picto <- prop.table(adv_picto_effectif)*100
interj_picto <- prop.table(interj_picto_effectif)*100
n_picto <- prop.table(n_picto_effectif)*100
pro_picto <- prop.table(pro_picto_effectif)*100
tp_picto <- prop.table(tp_picto_effectif)*100
v_picto <- prop.table(v_picto_effectif)*100
c(adj_picto["1"],adv_picto["1"],interj_picto["1"],n_picto["1"],pro_picto["1"],tp_picto["1"],v_picto["1"])
#34.28571 33.75000 27.77778 53.84615 42.07650 NA 51.45631 #t2p et pictodr
#36.36364 31.91489 33.33333 51.35135 43.68932 NA 42.27642 #t2p
#30.76923 36.36364 22.22222 57.14286 40.00000 NA 65.06024 #pictodr

#-----

#proportion de réponses correctes pour tous les pictos (#683) avec A1, par questionnaire
q1_picto <- prop.table(table(subset(df, questionnaire=="Q1", annotator1_picto_all)))*100
q2_picto <- prop.table(table(subset(df, questionnaire=="Q2", annotator1_picto_all)))*100
c(q1_picto["1"],q2_picto["1"]) #47.71084 36.94030

#proportion de réponses correctes pour toutes les phrases (#180) avec A1, par questionnaire
q1_sentence <- prop.table(table(subset(df_sentence, questionnaire=="Q1", annotator1_sentence)))*100
q2_sentence <- prop.table(table(subset(df_sentence, questionnaire=="Q2", annotator1_sentence)))*100
c(q1_sentence["1"],q2_sentence["1"]) #30 20