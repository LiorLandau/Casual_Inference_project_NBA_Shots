library(MASS)
model.data = read.csv ("shot_logs2.csv")
data=model.data[model.data$PTS_TYPE==3,]
three_pt=data$PTS_TYPE
three_pt <-three_pt[!is.na(three_pt)]
shot_clck=data$SHOT_CLOCK
shot_clck2=data$SHOT_CLOCK
shot_clck <-shot_clck[!is.na(shot_clck)]
hist(shot_clck,xlim = c(0,25),main="Histogram of Offense duration",col = 8)
FGM=data$FGM
#plot(data$SHOT_CLOCK,FGM)
log_reg <- glm(data$FGM ~data$SHOT_CLOCK ,family=binomial(link='logit'))
#log_reg <- glm(data$FGM ~.,data=data ,family=binomial(link='logit'))
summary(log_reg)
M=cbind(shot_clck2,FGM)
library(corrplot)
M=cor(M)
corrplot(M, method="circle")
# data2=data[c(5:23)]
# fit=glm(data2$FGM~. ,data=data2,family=binomial(link='logit'))
# summary(fit)
library("ggpubr")
res <- cor.test(data$FGM, data$SHOT_CLOCK, method = "pearson")
res
prop.table(table(data$FGM))
library(polycor)
polyserial(data$SHOT_CLOCK, data$FGM)
