install.packages("tree")
library(tree)
install.packages("pROC")
library(pROC)
install.packages("corrplot")
library("corrplot")
source(choose.files())
crypto_data <- read.csv(choose.files())
####################################################################################

##DATA CLEANING
#unique(crypto_data$crypto_name)
#dummy_crypto <- c("Bitcoin","Litecoin","XRP","Dogecoin","Monero","Stellar","Tether",
#           "Ethereum","Ethereum Classic","Maker","Basic Attention Token",
#           "EOS","Bitcoin Cash","BNB","TRON","Decentraland","Chainlink",
#           "Cardano","Filecoin","Theta Network","Huobi Token","Ravencoin",
#           "Tezos","VeChain","Quant","USD Coin","Cronos","Wrapped Bitcoin",
#           "Cosmos","Polygon","OKB","UNUS SED LEO","Algorand","Chiliz",
#           "THORChain","Terra Classic","FTX Token","Hedera","Binance USD",          
#           "Dai","Solana","Avalanche","Shiba Inu","The Sandbox","Polkadot",
#           "Elrond","Uniswap","Aave","NEAR Protocol","Flow","Internet Computer",    
#           "Casper","Toncoin","Chain","ApeCoin","Aptos")

#Looking at Bitcoin data and separating it into training and test according to timeline
crypto_Bitcoin <- crypto_data[crypto_data$crypto_name == "Bitcoin",]
crypto_Bitcoin_Train <- crypto_Bitcoin[crypto_Bitcoin$date < "2022-01-01",]
crypto_Bitcoin_Test <- crypto_Bitcoin[crypto_Bitcoin$date >= '2022-01-01',]

#Correlation plot for the variables
M = cor(crypto_Bitcoin)
corrplot(M)

#Creating dummy variables for all cryptocurrency names
#crypto_clean <- crypto_data
#crypto_clean$Bitcoin <- ifelse(crypto_data$crypto_name == "Bitcoin", 1, 0)
#crypto_clean$Litecoin <- ifelse(crypto_data$crypto_name == "Litecoin", 1, 0)
#crypto_clean$XRP <- ifelse(crypto_data$crypto_name == "XRP", 1, 0)
#crypto_clean$Dogecoin <- ifelse(crypto_data$crypto_name == "Dogecoin", 1, 0)
#crypto_clean$Monero <- ifelse(crypto_data$crypto_name == "Monero", 1, 0)
#crypto_clean$Stellar <- ifelse(crypto_data$crypto_name == "Stellar", 1, 0)
#crypto_clean$Tether <- ifelse(crypto_data$crypto_name == "Tether", 1, 0)
#crypto_clean$Ethereum <- ifelse(crypto_data$crypto_name == "Ethereum", 1, 0)
#crypto_clean$Ethereum_Classic <- ifelse(crypto_data$crypto_name == "Eth
#                                        ereum Classic", 1, 0)
#crypto_clean$Maker <- ifelse(crypto_data$crypto_name == "Maker", 1, 0)
#crypto_clean$Basic_Attention_Token <- ifelse(crypto_data$crypto_name == "Basic Attention Token", 1, 0)
#crypto_clean$EOS <- ifelse(crypto_data$crypto_name == "EOS", 1, 0)
#crypto_clean$Bitcoin_Cash<- ifelse(crypto_data$crypto_name == "Bitcoin Cash", 1, 0)
#crypto_clean$BNB <- ifelse(crypto_data$crypto_name == "BNB", 1, 0)
#crypto_clean$TRON <- ifelse(crypto_data$crypto_name == "TRON", 1, 0)
#crypto_clean$Decentraland <- ifelse(crypto_data$crypto_name == "Decentraland", 1, 0)
#crypto_clean$Chainlink <- ifelse(crypto_data$crypto_name == "Chainlink", 1, 0)
#crypto_clean$Cardano <- ifelse(crypto_data$crypto_name == "Cardano", 1, 0)
#crypto_clean$Filecoin <- ifelse(crypto_data$crypto_name == "Filecoin", 1, 0)
#crypto_clean$Theta_Network <- ifelse(crypto_data$crypto_name == "Theta Network", 1, 0)
#crypto_clean$Huobi_Token <- ifelse(crypto_data$crypto_name == "Huobi Token", 1, 0)
#crypto_clean$Ravecoin <- ifelse(crypto_data$crypto_name == "Ravecoin", 1, 0)
#crypto_clean$Tezos <- ifelse(crypto_data$crypto_name == "Tezos", 1, 0)
#crypto_clean$VeChain <- ifelse(crypto_data$crypto_name == "VeChain", 1, 0)
#crypto_clean$Quant <- ifelse(crypto_data$crypto_name == "Quant", 1, 0)
#crypto_clean$USD_Coin <- ifelse(crypto_data$crypto_name == "USD Coin", 1, 0)
#crypto_clean$Cronos <- ifelse(crypto_data$crypto_name == "Cronos", 1, 0)
#crypto_clean$Wrapped_Bitcoin <- ifelse(crypto_data$crypto_name == "Wrapped Bitcoin", 1, 0)
#crypto_clean$Cosmos <- ifelse(crypto_data$crypto_name == "Cosmos", 1, 0)
#crypto_clean$Polygon <- ifelse(crypto_data$crypto_name == "Polygon", 1, 0)
#crypto_clean$OKB <- ifelse(crypto_data$crypto_name == "OKB", 1, 0)
#crypto_clean$UNUS_SED_LEO <- ifelse(crypto_data$crypto_name == "UNUS SED LEO", 1, 0)
#crypto_clean$Algorand <- ifelse(crypto_data$crypto_name == "Algorand", 1, 0)
#crypto_clean$Chiliz <- ifelse(crypto_data$crypto_name == "Chiliz", 1, 0)
#crypto_clean$THORChain <- ifelse(crypto_data$crypto_name == "THORChain", 1, 0)
#crypto_clean$Terra_Classic <- ifelse(crypto_data$crypto_name == "Terra Classic", 1, 0)
#crypto_clean$FTX_Token <- ifelse(crypto_data$crypto_name == "FTX Token", 1, 0)
#crypto_clean$Hedera <- ifelse(crypto_data$crypto_name == "Hedera", 1, 0)
#crypto_clean$Binance_USD <- ifelse(crypto_data$crypto_name == "Binance USD", 1, 0)
#crypto_clean$Dai <- ifelse(crypto_data$crypto_name == "Dai", 1, 0)
#crypto_clean$Solana <- ifelse(crypto_data$crypto_name == "Solana", 1, 0)
#crypto_clean$Avalanche <- ifelse(crypto_data$crypto_name == "Avalanche", 1, 0)
#crypto_clean$Shiba_Inu <- ifelse(crypto_data$crypto_name == "Shiba Inu", 1, 0)
#crypto_clean$The_Sandbox <- ifelse(crypto_data$crypto_name == "The Sandbox", 1, 0)
#crypto_clean$Polkadot <- ifelse(crypto_data$crypto_name == "Polkadot", 1, 0)
#crypto_clean$Elrond <- ifelse(crypto_data$crypto_name == "Elrond", 1, 0)
#crypto_clean$Uniswap <- ifelse(crypto_data$crypto_name == "Uniswap", 1, 0)
#crypto_clean$Aave <- ifelse(crypto_data$crypto_name == "Aave", 1, 0)
#crypto_clean$NEAR_Protocol <- ifelse(crypto_data$crypto_name == "NEAR Protocol", 1, 0)
#crypto_clean$Flow <- ifelse(crypto_data$crypto_name == "Flow", 1, 0)
#crypto_clean$Internet_Computer <- ifelse(crypto_data$crypto_name == "Internet Computer", 1, 0)
#crypto_clean$Casper <- ifelse(crypto_data$crypto_name == "Casper", 1, 0)
#crypto_clean$Toncoin <- ifelse(crypto_data$crypto_name == "Toncoin", 1, 0)
#crypto_clean$Chain <- ifelse(crypto_data$crypto_name == "Chain", 1, 0)
#crypto_clean$Apecoin <- ifelse(crypto_data$crypto_name == "Apecoin", 1, 0)
#crypto_clean$Aptos <- ifelse(crypto_data$crypto_name == "Aptos", 1, 0)

#Creating Training and Testing data
#crypto_train <- crypto_clean[crypto_clean$date<"2022-01-01",]
#crypto_test <- crypto_clean[crypto_clean$date>="2022-01-01",]

#Dropping Crypto_name and timestamp columns
drop_crypto <- c("crypto_name","timestamp", "date")
#crypto_clean <- crypto_clean[,!(names(crypto_clean) %in% drop_crypto)]
#crypto_train <- crypto_train[,!(names(crypto_clean) %in% drop_crypto)]
#crypto_test <- crypto_test[,!(names(crypto_clean) %in% drop_crypto)]

crypto_Bitcoin <- crypto_Bitcoin[,!(names(crypto_Bitcoin) %in% drop_crypto)]
crypto_Bitcoin_Train <- crypto_Bitcoin_Train[,!(names(crypto_Bitcoin) %in% drop_crypto)]
crypto_Bitcoin_Test <- crypto_Bitcoin_Test[,!(names(crypto_Bitcoin) %in% drop_crypto)]

#DATA MODELING

#Data Model for predicting the close price (Regression)
installpkg("glmnet")
library(glmnet)

################ Model 1 : Linear regression
model_linear <- lm(close ~ . ,data = crypto_Bitcoin)
summary(model_linear)

# Linear regression with interaction + Lasso 
Mx<- model.matrix(close ~.^2, data = crypto_Bitcoin)[,-1]
My<- crypto_Bitcoin$close
lasso <- glmnet(Mx,My)
summary(lasso)
lassoCV <- cv.glmnet(Mx,My)
lassoCV$lambda.min
lassoCV$lambda.1se

################ Model 2 : Regression model with Lasso where lambda set to min
lassoMin <- glmnet(Mx,My,lambda = lassoCV$lambda.min)
summary(lassoMin)
support(lassoMin$beta)
lassoMin$beta
colnames(Mx)[support(lassoMin$beta)]
length(support(lassoMin$beta))

################ Model 3 : Regression model with Lasso where lambda set to 1se
lasso1se <- glmnet(Mx,My,lambda = lassoCV$lambda.1se)
summary(lasso1se)
support(lasso1se$beta)
colnames(Mx)[support(lasso1se$beta)]
length(support(lasso1se$beta))

# Ridge Regression
ridgeCV <- cv.glmnet(Mx,My, alpha=0)
ridgeCV$lambda.min
ridgeCV$lambda.1se

################ Model 4 : Regression model with Ridge where lambda set to min
ridgeMin <- glmnet(Mx,My, alpha = 0, lambda = ridgeCV$lambda.min)
summary(ridgeMin)
support(ridgeMin$beta)
colnames(Mx)[support(ridgeMin$beta)]
length(support(ridgeMin$beta))

################ Model 5 : Regression model with Ridge where lambda set to 1se
ridge1se <- glmnet(Mx,My, alpha = 0, lambda = ridgeCV$lambda.1se)
summary(ridge1se)
support(ridge1se$beta)
colnames(Mx)[support(ridge1se$beta)]
length(support(ridge1se$beta)) 


################ Model 6 : Random Forest
install.packages("randomForest")
library(randomForest)
library(ggplot2)

rf.fit <- randomForest(close ~ ., data=crypto_Bitcoin, ntree=1000,
                       keep.forest=FALSE, importance=TRUE)
set.seed(1234)
RF = rf.fit$rsq

################ Model 7 : XGBoost
#install.packages('xgboost')
#library('xgboost')

#define predictor and response variables in training set
#train_x = data.matrix(crypto_Bitcoin_Train[, -1])
#train_y = crypto_Bitcoin_Train[,1]

#define predictor and response variables in testing set
#test_x = data.matrix(crypto_Bitcoin_Test[, -1])
#test_y = crypto_Bitcoin_Test[, 1]

#define final training and testing sets
#xgb_train = xgb.DMatrix(data = train_x, label = train_y)
#xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#defining a watchlist
#watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteration
#model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)

#define final model
#model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)
#summary(model_xgboost)

#use model to make predictions on test data
#pred_y = predict(model_xgboost, xgb_test)

#cv <- xgb.cv(data = xgb_train, nrounds = 10, nthread = 2, nfold = 10, metrics = list("rmse","auc"),
#             max_depth = 3, eta = 1, objective = "reg:squaredlogerror")
#res = xgb.cv(data = xgb_train, nrounds = 10,num_boost_round=10, nfold=5,
#             metrics={'error'}, seed=0)

#param = {'max_depth':2, 'eta':1, 'objective':'reg:squarederror'}

#def fpreproc(xgb_train, xgb_test, param):
#  label = dtrain.get_label()
#  ratio = float(np.sum(label == 0)) / np.sum(label == 1)
#  param['scale_pos_weight'] = ratio
#  return (dtrain, dtest, param)
  
#xgb = xgb.cv(data = xgb_train, nrounds = 10, nfold=10,
#       metrics={'auc'}, seed=0, fpreproc=fpreproc)

#XGB.OOS = model_xgboost$evaluation_log

#Performance Metrics
set.seed(123)
nfold <- 10
n <- nrow(crypto_Bitcoin)
nreg<- nrow(crypto_Bitcoin)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
foldidreg <- rep(1:nfold,each=ceiling(nreg/nfold))[sample(1:nreg)]

# Perform cross validation on linear regression
LR <- c()
for(k in 1:nfold){ 
  train <- which(foldidreg!=k)
  reg <- lm(close ~., data = crypto_Bitcoin, subset = train)
  
  pred.reg <- predict(reg, newdata=crypto_Bitcoin[-train,], type="response")
  
  ## calculate and R2
  LR[k] <- R2(y=crypto_Bitcoin$close[-train], pred=pred.reg)
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

L.OOS <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold))
R.OOS <- data.frame(R.min=rep(NA,nfold), R.1se=rep(NA,nfold))

set.seed(123)
for(k in 1:nfold){
  train <- which(foldid!=k)
  ridgemin  <- glmnet(Mx[train,],My[train], alpha = 0, lambda = ridgeCV$lambda.min)
  ridge1se  <- glmnet(Mx[train,],My[train], alpha = 0, lambda = ridgeCV$lambda.1se)
  predridgemin <- predict(ridgemin, newx=Mx[-train,], type="response")
  predridge1se  <- predict(ridge1se, newx=Mx[-train,], type="response")
  R.OOS$R.min[k] <- R2(y=My[-train], pred=predridgemin)
  R.OOS$R.1se[k] <- R2(y=My[-train], pred=predridge1se)
  
  lassomin  <- glmnet(Mx[train,],My[train], lambda = lassoCV$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], lambda = lassoCV$lambda.1se)
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  
  L.OOS$L.min[k] <- R2(y=My[-train], pred=predlassomin)
  L.OOS$L.1se[k] <- R2(y=My[-train], pred=predlasso1se)
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}

#L.pred
R2performance <- cbind(L.OOS, R.OOS, LR,RF)
avgOOSR2 <- colMeans(R2performance)
barplot(colMeans(R2performance), las=2,xpd=FALSE, ylim=c(0,1) , xlab="", ylab = bquote( "Average Out of Sample " ~ R^2))
avgOOSR2

predictions <- predict(model_linear,crypto_Bitcoin_Test)
predictions
